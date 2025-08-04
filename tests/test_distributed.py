"""Tests for distributed utilities."""

import os

import pytest
import torch

from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_device,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
)


class TestDistributedUtilities:
    """Test distributed utilities in non-distributed mode."""

    def test_non_distributed_initialization(self):
        """Test that init_distributed works correctly when not in distributed mode."""
        # Clean environment
        env_vars = ["OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        rank, world_size, local_rank = init_distributed()

        assert rank == 0
        assert world_size == 1
        assert local_rank == 0

        # Cleanup shouldn't fail
        cleanup_distributed()

    def test_non_distributed_getters(self):
        """Test getter functions in non-distributed mode."""
        # Ensure we're not in distributed mode
        cleanup_distributed()

        assert get_rank() == 0
        assert get_world_size() == 1
        assert get_local_rank() == 0
        assert is_main_process()
        assert not is_distributed()

    def test_device_selection(self):
        """Test device selection in non-distributed mode."""
        device = get_device()

        if torch.cuda.is_available():
            assert device == "cuda"
        else:
            assert device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_with_local_rank(self):
        """Test device selection with local rank."""
        # Temporarily set LOCAL_RANK
        os.environ["LOCAL_RANK"] = "1"

        try:
            # In distributed mode, this would return cuda:1
            device = get_device()
            # But since we're not actually distributed, it returns cuda
            assert device == "cuda"
        finally:
            del os.environ["LOCAL_RANK"]


@pytest.mark.skip(reason="Requires actual distributed launch with mpirun")
class TestDistributedMode:
    """Tests that require actual distributed execution.

    These tests need to be run with:
    mpirun -np 2 pytest tests/test_distributed.py::TestDistributedMode
    """

    def test_distributed_initialization(self):
        """Test distributed initialization with MPI."""
        rank, world_size, local_rank = init_distributed()

        assert world_size == 2
        assert rank in [0, 1]
        assert local_rank in [0, 1]

        # Only rank 0 is main process
        assert is_main_process() == (rank == 0)
        assert is_distributed()

        cleanup_distributed()

    def test_data_distribution(self):
        """Test that each rank gets different data."""
        from spd.data import DatasetConfig, create_data_loader

        rank, world_size, _ = init_distributed()

        config = DatasetConfig(
            name="lennart-finke/SimpleStories",
            split="train[:100]",  # Use small subset
            n_ctx=128,
            is_tokenized=True,
        )

        loader, _ = create_data_loader(
            dataset_config=config,
            batch_size=4,
            buffer_size=100,
            global_seed=42,
            ddp_rank=rank,
            ddp_world_size=world_size,
        )

        # Get first batch from each rank
        _ = next(iter(loader))

        # TODO: Verify that different ranks get different data
        # This would require inter-process communication to compare

        cleanup_distributed()
