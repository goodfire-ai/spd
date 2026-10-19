from pathlib import Path

import pytest
from param_decomp_formats.datasets import RowGroupInfo, ShardInfo

from param_decomp.lm.batch_schedule import PrefixSchedule


def _shards(groups: tuple[tuple[int, ...], ...]) -> tuple[ShardInfo, ...]:
    """Shards whose row groups have the given row counts."""
    shards = []
    for index, counts in enumerate(groups):
        starts = [sum(counts[:i]) for i in range(len(counts))]
        shards.append(
            ShardInfo(
                Path(f"shard-{index}.parquet"),
                tuple(RowGroupInfo(start, n) for start, n in zip(starts, counts, strict=True)),
            )
        )
    return tuple(shards)


def _rows(schedule: PrefixSchedule) -> list[tuple[int, int]]:
    return [
        (selection.file_idx, int(row))
        for step in range(schedule.n_batches)
        for selection in schedule.batch_rows(step)
        for row in selection.rows
    ]


def test_a_prefix_is_the_datasets_first_rows_in_storage_order() -> None:
    schedule = PrefixSchedule(_shards(((3, 5), (0, 6))), 4, n_batches=3)
    assert _rows(schedule) == [(0, row) for row in range(8)] + [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert [len(schedule.batch_rows(step)) for step in range(3)] == [2, 1, 1]


def test_a_prefix_larger_than_the_dataset_is_refused() -> None:
    with pytest.raises(AssertionError, match="exceed the dataset"):
        PrefixSchedule(_shards(((4,), (4,))), 4, n_batches=3)


def test_steps_outside_the_prefix_are_refused() -> None:
    schedule = PrefixSchedule(_shards(((8,),)), 4, n_batches=2)
    for step in (-1, 2):
        with pytest.raises(IndexError, match="outside"):
            schedule.batch_rows(step)
