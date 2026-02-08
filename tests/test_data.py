"""Tests for index_split in create_data_loader.

Verifies that index_split correctly partitions data into disjoint train/val subsets
for streaming and non-streaming datasets, with and without DDP.
"""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, Features, IterableDataset, Sequence, Value
from torch.utils.data import DataLoader

from spd.data import DatasetConfig, create_data_loader
from spd.utils.distributed_utils import DistributedState

N_ROWS = 20
N_CTX = 4
BATCH_SIZE = 2
BUFFER_SIZE = 100
INDEX_SPLIT_N = 5

ALL_IDS = set(range(N_ROWS))
VAL_IDS = {i for i in range(N_ROWS) if i % INDEX_SPLIT_N == 0}
TRAIN_IDS = ALL_IDS - VAL_IDS


@pytest.fixture
def base_dataset() -> Dataset:
    """Tokenized dataset where row i has input_ids=[i, i, i, i]."""
    return Dataset.from_dict({"input_ids": [[i] * N_CTX for i in range(N_ROWS)]})


def _config(
    streaming: bool = False,
    index_split: tuple[int, Literal["train", "val"]] | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        name="fake",
        is_tokenized=True,
        hf_tokenizer_path="fake",
        streaming=streaming,
        split="train",
        n_ctx=N_CTX,
        seed=42,
        column_name="input_ids",
        shuffle_each_epoch=False,
        index_split=index_split,
    )


def _collect_row_ids(loader: DataLoader) -> set[int]:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    """Drain loader, return set of row identifiers (first element of each row)."""
    ids: set[int] = set()
    for batch in loader:
        ids.update(int(x) for x in batch["input_ids"][:, 0].tolist())
    return ids


def _make_iterable_dataset() -> IterableDataset:
    """Create a streaming dataset with proper features (needed for chained .filter() calls)."""

    def gen():
        for i in range(N_ROWS):
            yield {"input_ids": [i] * N_CTX}

    return IterableDataset.from_generator(
        gen, features=Features({"input_ids": Sequence(Value("int64"), length=N_CTX)})
    )


def _make_loader(  # pyright: ignore[reportUnknownParameterType]
    base_dataset: Dataset,
    streaming: bool = False,
    index_split: tuple[int, Literal["train", "val"]] | None = None,
    dist_state: DistributedState | None = None,
) -> DataLoader:  # pyright: ignore[reportMissingTypeArgument]
    ds: Dataset | IterableDataset = _make_iterable_dataset() if streaming else base_dataset
    cfg = _config(streaming=streaming, index_split=index_split)
    with (
        patch("spd.data.load_dataset", return_value=ds),
        patch("spd.data.AutoTokenizer.from_pretrained", return_value=MagicMock()),
    ):
        loader, _ = create_data_loader(
            dataset_config=cfg,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            dist_state=dist_state,
        )
    return loader


class TestIndexSplit:
    @pytest.mark.parametrize("streaming", [False, True])
    def test_no_split_returns_all(self, base_dataset: Dataset, streaming: bool) -> None:
        ids = _collect_row_ids(_make_loader(base_dataset, streaming=streaming))
        assert ids == ALL_IDS

    @pytest.mark.parametrize("streaming", [False, True])
    def test_val_split(self, base_dataset: Dataset, streaming: bool) -> None:
        ids = _collect_row_ids(
            _make_loader(base_dataset, streaming=streaming, index_split=(INDEX_SPLIT_N, "val"))
        )
        assert ids == VAL_IDS

    @pytest.mark.parametrize("streaming", [False, True])
    def test_train_split(self, base_dataset: Dataset, streaming: bool) -> None:
        ids = _collect_row_ids(
            _make_loader(base_dataset, streaming=streaming, index_split=(INDEX_SPLIT_N, "train"))
        )
        assert ids == TRAIN_IDS

    @pytest.mark.parametrize("streaming", [False, True])
    def test_train_val_disjoint_and_complete(
        self, base_dataset: Dataset, streaming: bool
    ) -> None:
        train_ids = _collect_row_ids(
            _make_loader(base_dataset, streaming=streaming, index_split=(INDEX_SPLIT_N, "train"))
        )
        val_ids = _collect_row_ids(
            _make_loader(base_dataset, streaming=streaming, index_split=(INDEX_SPLIT_N, "val"))
        )
        assert train_ids & val_ids == set()
        assert train_ids | val_ids == ALL_IDS

    def test_n_must_be_greater_than_1(self, base_dataset: Dataset) -> None:
        with pytest.raises(AssertionError, match="index_split N must be > 1"):
            _make_loader(base_dataset, index_split=(1, "val"))


class TestIndexSplitDDP:
    """Tests that index_split works correctly with DistributedState (simulated DDP)."""

    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("role", ["train", "val"])
    def test_all_ranks_cover_expected_split(
        self, base_dataset: Dataset, streaming: bool, role: str
    ) -> None:
        expected = VAL_IDS if role == "val" else TRAIN_IDS
        combined: set[int] = set()
        for rank in range(2):
            dist = DistributedState(rank=rank, world_size=2, local_rank=rank, backend="gloo")
            ids = _collect_row_ids(
                _make_loader(
                    base_dataset,
                    streaming=streaming,
                    index_split=(INDEX_SPLIT_N, role),  # pyright: ignore[reportArgumentType]
                    dist_state=dist,
                )
            )
            combined |= ids
        assert combined == expected

    @pytest.mark.parametrize("streaming", [False, True])
    def test_ranks_are_disjoint(self, base_dataset: Dataset, streaming: bool) -> None:
        per_rank: list[set[int]] = []
        for rank in range(2):
            dist = DistributedState(rank=rank, world_size=2, local_rank=rank, backend="gloo")
            ids = _collect_row_ids(
                _make_loader(
                    base_dataset,
                    streaming=streaming,
                    index_split=(INDEX_SPLIT_N, "val"),
                    dist_state=dist,
                )
            )
            per_rank.append(ids)
        assert per_rank[0] & per_rank[1] == set()
