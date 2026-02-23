"""Distributed tests for per-rank RNG seeding.

Verifies that:
- After seed_per_rank, torch random calls produce different values across ranks
- broadcast_model_params makes model parameters identical across ranks despite divergent RNG
- Different base seeds produce different per-rank seeds (no collisions)

This file can be run in two ways:

1. Directly with torchrun (fastest):
   torchrun --standalone --nproc_per_node=2 --master_port=29505 tests/test_seed_per_rank_distributed.py

2. Via pytest (runs torchrun in subprocess):
   pytest tests/test_seed_per_rank_distributed.py
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from spd.utils.distributed_utils import (
    broadcast_model_params,
    cleanup_distributed,
    gather_all_tensors,
    get_distributed_state,
    init_distributed,
    seed_per_rank,
    sync_across_processes,
)


def _test_rng_diverges_across_ranks():
    """After seed_per_rank, torch.randn produces different values on each rank."""
    state = get_distributed_state()
    assert state is not None

    seed_per_rank(42)
    samples = torch.randn(10)

    gathered = gather_all_tensors(samples)

    if state.rank == 0:
        assert not torch.allclose(gathered[0], gathered[1]), (
            "Random samples should differ across ranks after seed_per_rank"
        )
        print("  pass: RNG diverges across ranks")


def _test_same_seed_is_reproducible():
    """Calling seed_per_rank with the same base_seed twice gives the same RNG state."""
    state = get_distributed_state()
    assert state is not None

    seed_per_rank(42)
    samples_a = torch.randn(10)

    seed_per_rank(42)
    samples_b = torch.randn(10)

    assert torch.equal(samples_a, samples_b), "Same seed should produce same samples"
    if state.rank == 0:
        print("  pass: same seed is reproducible")


def _test_different_base_seeds_no_collision():
    """Different base_seed values produce different RNG streams on the same rank."""
    state = get_distributed_state()
    assert state is not None

    seed_per_rank(0)
    samples_seed0 = torch.randn(10)

    seed_per_rank(1)
    samples_seed1 = torch.randn(10)

    assert not torch.equal(samples_seed0, samples_seed1), (
        "Different base seeds should produce different samples on the same rank"
    )

    # Also check cross-rank: seed=0 on rank 1 should differ from seed=1 on rank 0
    gathered_seed0 = gather_all_tensors(samples_seed0)
    gathered_seed1 = gather_all_tensors(samples_seed1)

    if state.rank == 0:
        for i in range(state.world_size):
            for j in range(state.world_size):
                if i == j:
                    continue
                assert not torch.allclose(gathered_seed0[i], gathered_seed1[j]), (
                    f"seed=0/rank={i} should differ from seed=1/rank={j}"
                )
        print("  pass: different base seeds produce no collisions")


def _test_broadcast_model_params_syncs():
    """broadcast_model_params makes model params identical despite divergent RNG."""
    state = get_distributed_state()
    assert state is not None

    seed_per_rank(42)

    model = nn.Linear(8, 4)
    broadcast_model_params(model)

    weight_gathered = gather_all_tensors(model.weight.data)
    bias_gathered = gather_all_tensors(model.bias.data)

    if state.rank == 0:
        for r in range(1, state.world_size):
            torch.testing.assert_close(weight_gathered[0], weight_gathered[r])
            torch.testing.assert_close(bias_gathered[0], bias_gathered[r])
        print("  pass: broadcast_model_params syncs parameters")


def run_all_tests():
    init_distributed()
    try:
        state = get_distributed_state()
        assert state is not None
        assert state.world_size == 2, f"Tests require exactly 2 ranks, got {state.world_size}"

        tests = [
            ("rng diverges across ranks", _test_rng_diverges_across_ranks),
            ("same seed is reproducible", _test_same_seed_is_reproducible),
            ("different base seeds no collision", _test_different_base_seeds_no_collision),
            ("broadcast_model_params syncs", _test_broadcast_model_params_syncs),
        ]

        if state.rank == 0:
            print(f"\nRunning {len(tests)} seed_per_rank distributed tests...\n")

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                if state.rank == 0:
                    print(f"  FAIL: {test_name}: {e}")
                raise
            sync_across_processes()

        if state.rank == 0:
            print(f"\nAll {len(tests)} seed_per_rank distributed tests passed!\n")
    finally:
        cleanup_distributed()


@pytest.mark.slow
class TestSeedPerRank:
    def test_seed_per_rank_distributed(self):
        script_path = Path(__file__).resolve()

        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port",
            "29505",
            str(script_path),
        ]

        new_env = os.environ.copy()
        new_env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(cmd, env=new_env, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Distributed test failed with code {result.returncode}")

        print(result.stderr)


if __name__ == "__main__":
    run_all_tests()
