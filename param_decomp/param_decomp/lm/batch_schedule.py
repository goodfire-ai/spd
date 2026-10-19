"""Deterministic row selection from dataset metadata, without reading dataset contents."""

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from param_decomp_formats.datasets import ShardInfo


@dataclass(frozen=True)
class RowGroupSelection:
    """Absolute shard row indices to read together from one physical row group."""

    file_idx: int
    row_group_idx: int
    rows: NDArray[np.int64]


type BatchRowIndices = tuple[RowGroupSelection, ...]


class BatchRowSchedule(Protocol):
    @property
    def shards(self) -> tuple[ShardInfo, ...]: ...

    @property
    def global_batch(self) -> int: ...

    def batch_rows(self, step: int) -> BatchRowIndices: ...


@dataclass(frozen=True)
class RowStream:
    """Row groups `groups` (file index, row-group index) read one after another, the group at
    position i ending at stream row `row_ends[i]`."""

    groups: tuple[tuple[int, int], ...]
    row_ends: tuple[int, ...]

    @staticmethod
    def of(shards: tuple[ShardInfo, ...], groups: Sequence[tuple[int, int]]) -> "RowStream":
        row_ends = np.cumsum([shards[f].row_groups[g].n_rows for f, g in groups]).tolist()
        return RowStream(tuple(groups), tuple(row_ends))

    def cut(
        self,
        shards: tuple[ShardInfo, ...],
        start: int,
        stop: int,
        group_rows: Callable[[int, int], NDArray[np.int64]],
    ) -> BatchRowIndices:
        """Stream rows `start..stop`, as selections of absolute shard rows; `group_rows(f, g)`
        is the order a group's rows are read in, as offsets into the group."""
        assert 0 <= start < stop <= self.row_ends[-1], (start, stop, self.row_ends[-1])
        position = bisect_right(self.row_ends, start)
        selections = []
        while start < stop:
            group_start = 0 if position == 0 else self.row_ends[position - 1]
            end = min(stop, self.row_ends[position])
            file_idx, row_group_idx = self.groups[position]
            group = shards[file_idx].row_groups[row_group_idx]
            rows = group_rows(file_idx, row_group_idx)[start - group_start : end - group_start]
            selections.append(RowGroupSelection(file_idx, row_group_idx, group.start_row + rows))
            start = end
            position += 1
        return tuple(selections)


def _perm(seed_parts: tuple[int, ...], n: int) -> NDArray[np.int64]:
    return np.random.default_rng(seed_parts).permutation(n)


class BatchSchedule:
    """Fixed-size batches of a row-shuffled dataset
    (`param_decomp_formats.dataset_store.assert_row_shuffled`).
    An epoch's stream is every row group once, in an order seeded by the epoch, each group's
    rows in an order seeded by the epoch and group; its batches are consecutive full
    `global_batch`-row cuts of that stream, and the final `n_rows % global_batch` rows are
    not read that epoch.

    A step determines its rows independently of lookup history; only the epoch's group order
    is cached."""

    def __init__(self, shards: tuple[ShardInfo, ...], global_batch: int, seed: int):
        assert shards and all(shard.n_rows > 0 for shard in shards)
        assert global_batch > 0
        self.shards = shards
        self.global_batch = global_batch
        self.seed = seed
        self.steps_per_epoch = sum(shard.n_rows for shard in shards) // global_batch
        assert self.steps_per_epoch > 0, f"global_batch={global_batch} larger than the corpus"
        self._groups = [
            (file_idx, group_idx)
            for file_idx, shard in enumerate(shards)
            for group_idx, group in enumerate(shard.row_groups)
            if group.n_rows
        ]
        self._cached: tuple[int, RowStream] | None = None

    def _epoch(self, epoch: int) -> RowStream:
        if self._cached is None or self._cached[0] != epoch:
            order = _perm((self.seed, epoch), len(self._groups))
            self._cached = (epoch, RowStream.of(self.shards, [self._groups[i] for i in order]))
        return self._cached[1]

    def batch_rows(self, step: int) -> BatchRowIndices:
        assert step >= 0
        epoch, batch_in_epoch = divmod(step, self.steps_per_epoch)
        start = batch_in_epoch * self.global_batch
        return self._epoch(epoch).cut(
            self.shards,
            start,
            start + self.global_batch,
            lambda f, g: _perm((self.seed, epoch, f, g), self.shards[f].row_groups[g].n_rows),
        )


def slice_batch_rows(rows: BatchRowIndices, start: int, stop: int) -> BatchRowIndices:
    """Take an interval in batch order, preserving physical read boundaries."""
    assert 0 <= start <= stop
    offset = 0
    selected = []
    for selection in rows:
        end = offset + len(selection.rows)
        lo, hi = max(start, offset), min(stop, end)
        if lo < hi:
            selected.append(
                RowGroupSelection(
                    selection.file_idx,
                    selection.row_group_idx,
                    selection.rows[lo - offset : hi - offset],
                )
            )
        offset = end
    assert stop <= offset, (stop, offset)
    return tuple(selected)


class PrefixSchedule:
    """The first `n_batches` batches of a row-shuffled dataset, in storage order: batch
    `step` is rows `step * global_batch ...`, read shard by shard and row group by row group.
    Any prefix of such a dataset is a uniform sample of it."""

    def __init__(self, shards: tuple[ShardInfo, ...], global_batch: int, n_batches: int):
        assert shards and global_batch > 0 and n_batches > 0
        self.shards = shards
        self.global_batch = global_batch
        self.n_batches = n_batches
        self._stream = RowStream.of(
            shards,
            [
                (file_idx, group_idx)
                for file_idx, shard in enumerate(shards)
                for group_idx, group in enumerate(shard.row_groups)
                if group.n_rows
            ],
        )
        assert n_batches * global_batch <= self._stream.row_ends[-1], (
            f"{n_batches} batches of {global_batch} exceed the dataset's "
            f"{self._stream.row_ends[-1]} rows"
        )

    def batch_rows(self, step: int) -> BatchRowIndices:
        if not 0 <= step < self.n_batches:
            raise IndexError(f"batch {step} outside the prefix's [0, {self.n_batches})")
        start = step * self.global_batch
        return self._stream.cut(
            self.shards,
            start,
            start + self.global_batch,
            lambda f, g: np.arange(self.shards[f].row_groups[g].n_rows),
        )
