"""Tests for distributed utilities."""

import pytest
import torch

from spd.utils import distributed_utils
from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
)


class TestDistributedUtilities:
    """Test distributed utilities in non-distributed mode."""

    def test_non_distributed_getters(self):
        """Test getter functions in non-distributed mode."""
        # Ensure we're not in distributed mode
        cleanup_distributed()

        assert get_rank() == 0
        assert get_world_size() == 1
        assert get_local_rank() == 0
        assert is_main_process()
        assert not is_distributed()

    @pytest.mark.parametrize(
        "cuda_available, distributed, local_rank, expected",
        [
            (False, False, 0, "cpu"),
            (False, True, 1, "cpu"),
            (True, False, 0, "cuda"),
            (True, True, 0, "cuda:0"),
            (True, True, 2, "cuda:2"),
        ],
    )
    def test_get_device_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cuda_available: bool,
        distributed: bool,
        local_rank: int,
        expected: str,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available, raising=False)
        monkeypatch.setattr(distributed_utils, "is_distributed", lambda: distributed)
        monkeypatch.setattr(distributed_utils, "get_local_rank", lambda: local_rank)
        assert distributed_utils.get_device() == expected


@pytest.mark.skip(reason="Requires actual distributed launch with mpirun")
class TestDistributedMode:
    """Tests that require actual distributed execution.

    These tests need to be run with:
    mpirun -np 2 pytest tests/test_distributed.py::TestDistributedMode
    """

    def test_distributed_initialization(self):
        """Test distributed initialization with MPI."""
        dist_state = init_distributed()

        assert dist_state.world_size == 2
        assert dist_state.rank in [0, 1]
        assert dist_state.local_rank in [0, 1]

        # Only rank 0 is main process
        assert is_main_process() == (dist_state.rank == 0)
        assert is_distributed()

        cleanup_distributed()

    def test_data_distribution(self):
        """Test that each rank gets different data."""
        from spd.data import DatasetConfig, create_data_loader

        dist_state = init_distributed()

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
            ddp_rank=dist_state.rank,
            ddp_world_size=dist_state.world_size,
        )

        # Get first batch from each rank
        _ = next(iter(loader))

        # TODO: Verify that different ranks get different data
        # This would require inter-process communication to compare

        cleanup_distributed()
