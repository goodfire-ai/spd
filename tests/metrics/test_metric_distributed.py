"""Distributed tests for custom Metric class.

This file can be run in two ways:

1. Directly with mpirun (fastest):
   mpirun -np 2 python tests/test_metric_distributed.py

2. Via pytest (runs mpirun in subprocess):
   pytest tests/test_metric_distributed.py
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, override

import pytest
import torch
import torch.distributed as dist

from spd.metrics.base import Metric


class SumMetric(Metric):
    """Simple metric that computes mean using sum reduction."""

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, *, value: torch.Tensor, **_: Any) -> None:
        self.total += value.sum()
        self.count += value.numel()

    @override
    def compute(self) -> float:
        return (self.total / self.count).item()


class CatMetric(Metric):
    """Simple metric that concatenates values using cat reduction."""

    values: list[torch.Tensor] | torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    @override
    def update(self, *, value: torch.Tensor, **_: Any) -> None:
        assert isinstance(self.values, list)
        self.values.append(value)

    @override
    def compute(self) -> torch.Tensor:
        if isinstance(self.values, list):
            return torch.cat(self.values, dim=0)
        return self.values


def init_distributed():
    """Initialize distributed backend for testing."""
    if not dist.is_available():
        raise RuntimeError("Distributed not available")

    # Use environment variables set by mpirun
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        raise RuntimeError("No distributed environment detected")

    # Set required environment variables for PyTorch
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    # Initialize process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _test_sum_metric_sync():
    """Test that sum metric correctly syncs across ranks."""
    rank, _world_size = init_distributed()

    try:
        metric = SumMetric()

        # Each rank updates with different values
        # Rank 0: [0, 1], Rank 1: [2, 3]
        values = torch.tensor([rank * 2.0, rank * 2.0 + 1.0])
        metric.update(value=values)

        # Before sync: each rank has its own local state
        local_mean = metric.compute()
        if rank == 0:
            expected_local = (0 + 1) / 2
            assert abs(local_mean - expected_local) < 1e-6, (
                f"Rank {rank}: {local_mean} != {expected_local}"
            )
        elif rank == 1:
            expected_local = (2 + 3) / 2
            assert abs(local_mean - expected_local) < 1e-6, (
                f"Rank {rank}: {local_mean} != {expected_local}"
            )

        # Sync across ranks
        metric.sync_dist()

        # After sync: all ranks should have global mean
        global_mean = metric.compute()
        expected_global = (0 + 1 + 2 + 3) / 4  # All values from both ranks
        assert abs(global_mean - expected_global) < 1e-6, (
            f"Rank {rank}: global_mean={global_mean} != expected={expected_global}"
        )

        if rank == 0:
            print(f"✓ Sum metric sync test passed (global_mean={global_mean:.4f})")

    finally:
        cleanup_distributed()


def _test_cat_metric_sync():
    """Test that cat metric correctly syncs across ranks."""
    rank, _world_size = init_distributed()

    try:
        metric = CatMetric()

        # Each rank updates with different values
        # Rank 0: [0, 1], Rank 1: [2, 3]
        values = torch.tensor([rank * 2.0, rank * 2.0 + 1.0])
        metric.update(value=values)

        # Before sync: each rank has its own local values
        local_values = metric.compute()
        if rank == 0:
            torch.testing.assert_close(local_values, torch.tensor([0.0, 1.0]))
        elif rank == 1:
            torch.testing.assert_close(local_values, torch.tensor([2.0, 3.0]))

        # Sync across ranks
        metric.sync_dist()

        # After sync: all ranks should have all values from all ranks
        global_values = metric.compute()
        # Values should be concatenated: rank 0's values, then rank 1's values
        expected_global = torch.tensor([0.0, 1.0, 2.0, 3.0])
        torch.testing.assert_close(global_values, expected_global)

        if rank == 0:
            print(f"✓ Cat metric sync test passed (global_values={global_values.tolist()})")

    finally:
        cleanup_distributed()


def _test_multiple_updates_then_sync():
    """Test multiple updates followed by sync."""
    rank, _world_size = init_distributed()

    try:
        metric = SumMetric()

        # Multiple updates per rank
        for i in range(3):
            value = torch.tensor([rank * 10.0 + i])
            metric.update(value=value)

        # Sync
        metric.sync_dist()

        # Compute global mean
        global_mean = metric.compute()

        # Rank 0 values: [0, 1, 2]
        # Rank 1 values: [10, 11, 12]
        # Global mean: (0+1+2+10+11+12) / 6 = 36 / 6 = 6
        expected_global = 6.0
        assert abs(global_mean - expected_global) < 1e-6, (
            f"Rank {rank}: global_mean={global_mean} != expected={expected_global}"
        )

        if rank == 0:
            print(f"✓ Multiple updates test passed (global_mean={global_mean:.4f})")

    finally:
        cleanup_distributed()


def _test_empty_cat_metric_sync():
    """Test that empty cat metric doesn't crash on sync."""
    rank, _world_size = init_distributed()

    try:
        metric = CatMetric()

        # Don't update - leave empty

        # Sync should not crash even with empty state
        metric.sync_dist()

        # Still empty after sync
        values = metric.values
        assert isinstance(values, list)
        assert len(values) == 0

        if rank == 0:
            print("✓ Empty cat metric sync test passed")

    finally:
        cleanup_distributed()


def run_all_tests():
    """Run all distributed tests when called directly with mpirun."""
    tests = [
        ("Sum metric sync", _test_sum_metric_sync),
        ("Cat metric sync", _test_cat_metric_sync),
        ("Multiple updates then sync", _test_multiple_updates_then_sync),
        ("Empty cat metric sync", _test_empty_cat_metric_sync),
    ]

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))

    if rank == 0:
        print(f"\nRunning {len(tests)} distributed metric tests...\n")

    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            if rank == 0:
                print(f"✗ {test_name} failed: {e}")
            raise
        # Small barrier to ensure clean test separation
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        print(f"\n✓ All {len(tests)} distributed tests passed!\n")


# ===== Pytest wrapper =====
# This allows running via pytest, which will spawn mpirun in a subprocess
@pytest.mark.slow
class TestDistributedMetrics:
    """Pytest wrapper for distributed metric tests."""

    def test_distributed_metrics(self):
        """Run distributed tests via mpirun in subprocess."""
        script_path = Path(__file__).resolve()

        env = {
            "MASTER_PORT": "29500",
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

        # Print output for visibility
        print(result.stdout)


if __name__ == "__main__":
    # When run directly with mpirun, execute all tests
    run_all_tests()
