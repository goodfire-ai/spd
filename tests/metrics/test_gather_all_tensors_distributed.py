"""Distributed tests for _gather_all_tensors function.

This file can be run in two ways:

1. Directly with mpirun (fastest):
   mpirun -np 2 python tests/metrics/test_gather_all_tensors_distributed.py

2. Via pytest (runs mpirun in subprocess):
   pytest tests/metrics/test_gather_all_tensors_distributed.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from spd.metrics.base import _gather_all_tensors


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


def _test_gather_identical_shapes():
    """Test gathering tensors with identical shapes across ranks."""
    rank, world_size = init_distributed()

    try:
        # Each rank has a different tensor with same shape
        tensor = torch.tensor([rank * 1.0, rank * 2.0])

        # Gather from all ranks
        gathered = _gather_all_tensors(tensor)

        # Should have one tensor per rank
        assert len(gathered) == world_size

        # Check shapes are all identical
        for t in gathered:
            assert t.shape == tensor.shape

        # Check values
        for i, t in enumerate(gathered):
            expected = torch.tensor([i * 1.0, i * 2.0])
            torch.testing.assert_close(t, expected)

        # Verify that our rank's entry is the original tensor (preserves autograd)
        assert gathered[rank] is tensor

        if rank == 0:
            print("✓ Gather identical shapes test passed")

    finally:
        cleanup_distributed()


def _test_gather_scalar_tensors():
    """Test gathering scalar tensors."""
    rank, world_size = init_distributed()

    try:
        # Scalar tensor with rank-specific value
        tensor = torch.tensor(rank * 10.0)

        # Gather from all ranks
        gathered = _gather_all_tensors(tensor)

        # Should have one tensor per rank
        assert len(gathered) == world_size

        # Check values
        for i, t in enumerate(gathered):
            expected = torch.tensor(i * 10.0)
            torch.testing.assert_close(t, expected)

        if rank == 0:
            print("✓ Gather scalar tensors test passed")

    finally:
        cleanup_distributed()


def _test_gather_multidimensional_tensors():
    """Test gathering multidimensional tensors."""
    rank, world_size = init_distributed()

    try:
        # 2D tensor with rank-specific values
        tensor = torch.full((3, 4), fill_value=float(rank))

        # Gather from all ranks
        gathered = _gather_all_tensors(tensor)

        # Should have one tensor per rank
        assert len(gathered) == world_size

        # Check shapes
        for t in gathered:
            assert t.shape == (3, 4)

        # Check values
        for i, t in enumerate(gathered):
            expected = torch.full((3, 4), fill_value=float(i))
            torch.testing.assert_close(t, expected)

        if rank == 0:
            print("✓ Gather multidimensional tensors test passed")

    finally:
        cleanup_distributed()


def _test_gather_empty_tensor():
    """Test gathering empty tensors."""
    rank, _world_size = init_distributed()

    try:
        # Empty tensor with consistent shape
        tensor = torch.tensor([])

        # Gather from all ranks
        gathered = _gather_all_tensors(tensor)

        # All gathered tensors should be empty
        for t in gathered:
            assert t.numel() == 0
            assert t.shape == tensor.shape

        if rank == 0:
            print("✓ Gather empty tensor test passed")

    finally:
        cleanup_distributed()


def _test_gather_large_tensor():
    """Test gathering larger tensors."""
    rank, world_size = init_distributed()

    try:
        # Larger tensor with rank-specific pattern
        tensor = torch.arange(1000, dtype=torch.float32) + rank * 1000

        # Gather from all ranks
        gathered = _gather_all_tensors(tensor)

        # Should have one tensor per rank
        assert len(gathered) == world_size

        # Check values
        for i, t in enumerate(gathered):
            expected = torch.arange(1000, dtype=torch.float32) + i * 1000
            torch.testing.assert_close(t, expected)

        if rank == 0:
            print("✓ Gather large tensor test passed")

    finally:
        cleanup_distributed()


def _test_gather_preserves_autograd():
    """Test that gathered tensor for current rank preserves autograd."""
    rank, _world_size = init_distributed()

    try:
        # Tensor with gradient tracking
        tensor = torch.tensor([rank * 1.0, rank * 2.0], requires_grad=True)

        # Gather from all ranks
        gathered = _gather_all_tensors(tensor)

        # Our rank's tensor should be the original (preserving autograd)
        assert gathered[rank] is tensor
        assert gathered[rank].requires_grad

        if rank == 0:
            print("✓ Gather preserves autograd test passed")

    finally:
        cleanup_distributed()


def _test_gather_non_initialized():
    """Test that gather works correctly when distributed is not initialized."""
    # Don't initialize distributed
    tensor = torch.tensor([1.0, 2.0, 3.0])

    # Should return single-element list with the tensor
    gathered = _gather_all_tensors(tensor)

    assert len(gathered) == 1
    torch.testing.assert_close(gathered[0], tensor)

    print("✓ Gather non-initialized test passed")


def run_all_tests():
    """Run all distributed tests when called directly with mpirun."""
    tests = [
        ("Gather identical shapes", _test_gather_identical_shapes),
        ("Gather scalar tensors", _test_gather_scalar_tensors),
        ("Gather multidimensional tensors", _test_gather_multidimensional_tensors),
        ("Gather empty tensor", _test_gather_empty_tensor),
        ("Gather large tensor", _test_gather_large_tensor),
        ("Gather preserves autograd", _test_gather_preserves_autograd),
    ]

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))

    if rank == 0:
        print(f"\nRunning {len(tests)} _gather_all_tensors tests...\n")

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

    # Test non-distributed case separately (only on rank 0)
    if rank == 0:
        # Need to clean up distributed first
        cleanup_distributed()
        _test_gather_non_initialized()


# ===== Pytest wrapper =====
# This allows running via pytest, which will spawn mpirun in a subprocess
@pytest.mark.slow
class TestGatherAllTensors:
    """Pytest wrapper for _gather_all_tensors tests."""

    def test_gather_all_tensors_distributed(self):
        """Run distributed tests via mpirun in subprocess."""
        script_path = Path(__file__).resolve()

        env = {
            "MASTER_PORT": "29501",
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
