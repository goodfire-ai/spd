"""Distributed tests for AliveComponentsTracker metric.

Run with: mpirun -np 2 python tests/metrics/test_alive_components_distributed.py
Or via pytest (slower): pytest tests/metrics/test_alive_components_distributed.py --runslow
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from spd.metrics.alive_components import AliveComponentsTracker
from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    sync_across_processes,
    with_distributed_cleanup,
)


def _test_min_reduction():
    """Test that compute() uses min reduction correctly."""
    rank = get_rank()

    metric = AliveComponentsTracker(
        module_paths=["layer1"],
        C=3,
        device="cpu",
        n_examples_until_dead=100,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=2,
    )

    # Initialize n_batches_until_dead by calling update once
    # CI shape (3,) = 1 example per rank * 2 ranks = 2 global examples
    # n_batches_until_dead = 100 // 2 = 50
    metric.update(ci={"layer1": torch.tensor([0.0, 0.0, 0.0])})

    # Set different counter values on each rank
    if rank == 0:
        metric.n_batches_since_fired["layer1"] = torch.tensor([5, 2, 8])
    else:
        metric.n_batches_since_fired["layer1"] = torch.tensor([3, 4, 1])

    # compute() will sync and apply min reduction
    # After min reduction: min(5,3)=3 < 50, min(2,4)=2 < 50, min(8,1)=1 < 50
    # All components should be alive
    result = metric.compute()
    assert result["n_alive/layer1"] == 3

    if rank == 0:
        print("✓ Min reduction test passed")


def _test_different_firing_patterns():
    """Test that components firing on any rank are considered alive globally."""
    rank = get_rank()

    metric = AliveComponentsTracker(
        module_paths=["layer1"],
        C=3,
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=2,
    )

    # Run 3 batches with different firing on each rank
    for _ in range(3):
        if rank == 0:
            # Rank 0: only component 0 fires
            ci = {"layer1": torch.tensor([0.2, 0.0, 0.0])}
        else:
            # Rank 1: only component 1 fires
            ci = {"layer1": torch.tensor([0.0, 0.2, 0.0])}
        metric.update(ci=ci)

    # Before compute: each rank has different local state
    if rank == 0:
        assert metric.n_batches_since_fired["layer1"][0] == 0  # fired locally
        assert metric.n_batches_since_fired["layer1"][1] == 3  # didn't fire locally
        assert metric.n_batches_since_fired["layer1"][2] == 3  # didn't fire locally
    else:
        assert metric.n_batches_since_fired["layer1"][0] == 3  # didn't fire locally
        assert metric.n_batches_since_fired["layer1"][1] == 0  # fired locally
        assert metric.n_batches_since_fired["layer1"][2] == 3  # didn't fire locally

    # compute() will sync with min reduction:
    # Component 0: min(0, 3) = 0 (fired on rank 0)
    # Component 1: min(3, 0) = 0 (fired on rank 1)
    # Component 2: min(3, 3) = 3 (didn't fire on either)
    # n_batches_until_dead = 50 // (1 * 2) = 25
    # All < 25, so all alive
    result = metric.compute()
    assert result["n_alive/layer1"] == 3  # all components alive

    if rank == 0:
        print(f"✓ Different firing patterns test passed (n_alive={result['n_alive/layer1']})")


def _test_dead_components():
    """Test that components are correctly marked as dead after threshold."""
    rank = get_rank()

    metric = AliveComponentsTracker(
        module_paths=["layer1"],
        C=3,
        device="cpu",
        n_examples_until_dead=5,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=2,
    )

    # Run batches where component 2 never fires
    # CI shape is (3,) which is [C], so 1 local example * 2 ranks = 2 global examples per batch
    # n_batches_until_dead = 5 // 2 = 2
    for _ in range(3):  # Need 3 batches to exceed threshold (3*2=6 > 5)
        if rank == 0:
            ci = {"layer1": torch.tensor([0.2, 0.0, 0.0])}
        else:
            ci = {"layer1": torch.tensor([0.0, 0.2, 0.0])}
        metric.update(ci=ci)

    # compute() will sync with min reduction:
    # Component 0: min(0, 3) = 0 (alive)
    # Component 1: min(3, 0) = 0 (alive)
    # Component 2: min(3, 3) = 3 (dead, >= 2)
    print(f"Rank {rank} n_batches_since_fired: {metric.n_batches_since_fired['layer1']}")
    result = metric.compute()
    # only components 0 and 1 alive
    assert result["n_alive/layer1"] == 2, (
        f"Expected 2 alive components, got {result['n_alive/layer1']}"
    )

    if rank == 0:
        print(f"✓ Dead components test passed (n_alive={result['n_alive/layer1']})")


def _test_multiple_modules():
    """Test tracking across multiple modules in distributed setting."""
    rank = get_rank()

    metric = AliveComponentsTracker(
        module_paths=["layer1", "layer2"],
        C=2,
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=2,
    )

    # Each rank fires different components in different modules
    if rank == 0:
        ci = {
            "layer1": torch.tensor([0.2, 0.0]),
            "layer2": torch.tensor([0.0, 0.0]),
        }
    else:
        ci = {
            "layer1": torch.tensor([0.0, 0.0]),
            "layer2": torch.tensor([0.0, 0.2]),
        }

    metric.update(ci=ci)

    # compute() will sync with min reduction:
    # layer1: min(0, 1) = 0, min(1, 1) = 1
    # layer2: min(1, 1) = 1, min(1, 0) = 0
    # n_batches_until_dead = 50 // (1 * 2) = 25
    # All < 25, so all alive
    result = metric.compute()
    assert result["n_alive/layer1"] == 2
    assert result["n_alive/layer2"] == 2

    if rank == 0:
        print(
            f"✓ Multiple modules test passed "
            f"(layer1={result['n_alive/layer1']}, layer2={result['n_alive/layer2']})"
        )


@with_distributed_cleanup
def run_all_tests():
    """Run all distributed tests when called directly with mpirun."""
    init_distributed(backend="gloo")
    rank = get_rank()
    world_size = get_world_size()

    if world_size != 2:
        if rank == 0:
            print(f"✗ Tests require exactly 2 ranks, got {world_size}")
        cleanup_distributed()
        sys.exit(1)

    tests = [
        ("Min reduction", _test_min_reduction),
        ("Different firing patterns", _test_different_firing_patterns),
        ("Dead components", _test_dead_components),
        ("Multiple modules", _test_multiple_modules),
    ]

    if rank == 0:
        print(f"\nRunning {len(tests)} distributed AliveComponentsTracker tests...\n")

    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            if rank == 0:
                print(f"✗ {test_name} failed: {e}")
            raise
        # Barrier to ensure clean test separation
        sync_across_processes()

    if rank == 0:
        print(f"\n✓ All {len(tests)} distributed tests passed!\n")


# ===== Pytest wrapper =====
@pytest.mark.slow
class TestDistributedAliveComponentsTracker:
    """Pytest wrapper for distributed AliveComponentsTracker tests."""

    def test_distributed_alive_components(self):
        """Run distributed tests via mpirun in subprocess."""
        script_path = Path(__file__).resolve()

        env = {
            "MASTER_PORT": "29503",
            "OMP_NUM_THREADS": "1",
        }

        cmd = ["mpirun", "-np", "2", sys.executable, str(script_path)]

        result = subprocess.run(
            cmd, env={**os.environ, **env}, capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Distributed test failed with code {result.returncode}")

        print(result.stdout)


if __name__ == "__main__":
    run_all_tests()
