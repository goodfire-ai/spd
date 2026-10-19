"""Read planned Parquet rows into process-local batches and place them on devices."""

from dataclasses import dataclass
from typing import Protocol

import jax
import numpy as np
import pyarrow.parquet as pq
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.typing import NDArray
from param_decomp_formats.datasets import ShardInfo, read_id_column, read_row_group
from param_decomp_formats.packed_rows import PackedRows

from param_decomp.core.placement import batch_axes
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.batch_schedule import (
    BatchRowIndices,
    BatchRowSchedule,
    RowGroupSelection,
    slice_batch_rows,
)
from param_decomp.sequence import SequenceLayout


def packed_rows_on_device(rows: PackedRows, vocab_size: int) -> LMBatchWithDocuments:
    """`rows` on the default device, their tokens checked against `vocab_size`."""
    rows.validate_vocabulary(vocab_size)
    return LMBatchWithDocuments(
        LMBatch(jax.device_put(rows.token_ids)),
        SequenceLayout(jax.device_put(rows.document_ids)),
    )


def global_lm_batch(
    local: PackedRows, mesh: Mesh, global_batch: int, vocab_size: int
) -> LMBatchWithDocuments:
    """Place both packed arrays on the same global batch axes after host validation."""
    local.validate_vocabulary(vocab_size)
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    shape = (global_batch, local.token_ids.shape[1])
    return LMBatchWithDocuments(
        LMBatch(jax.make_array_from_process_local_data(sharding, local.token_ids, shape)),
        SequenceLayout(jax.make_array_from_process_local_data(sharding, local.document_ids, shape)),
    )


def _local_batch_rows(
    schedule: BatchRowSchedule, step: int, process_index: int, process_count: int
) -> BatchRowIndices:
    rows = schedule.batch_rows(step)
    assert sum(len(selection.rows) for selection in rows) == schedule.global_batch
    per_process = schedule.global_batch // process_count
    start = process_index * per_process
    return slice_batch_rows(rows, start, start + per_process)


class ShardServer:
    """Read scheduled row groups and serve this process's global-batch slice."""

    def __init__(
        self,
        schedule: BatchRowSchedule,
        seq_len: int,
        process_index: int,
        process_count: int,
    ):
        assert process_count > 0 and 0 <= process_index < process_count
        assert seq_len > 0
        assert schedule.global_batch % process_count == 0, (
            f"global_batch={schedule.global_batch} not divisible by {process_count} processes"
        )
        self.schedule = schedule
        self.seq_len = seq_len
        self.process_index = process_index
        self.process_count = process_count
        self._loaded_row_group: tuple[int, int] | None = None
        self._batch: PackedRows | None = None

    @property
    def per_process(self) -> int:
        return self.schedule.global_batch // self.process_count

    def _load_row_group(self, loc: RowGroupSelection) -> PackedRows:
        key = (loc.file_idx, loc.row_group_idx)
        if self._loaded_row_group != key:
            self._batch = read_row_group(
                self.schedule.shards[loc.file_idx], loc.row_group_idx, self.seq_len
            )
            self._loaded_row_group = key
        assert self._batch is not None
        return self._batch

    def local_batch(self, step: int) -> PackedRows:
        """This process's paired `[per_process, seq_len]` int32 rows."""
        tokens, documents = [], []
        for selection in _local_batch_rows(
            self.schedule, step, self.process_index, self.process_count
        ):
            group = self.schedule.shards[selection.file_idx].row_groups[selection.row_group_idx]
            batch = self._load_row_group(selection)
            rows = selection.rows - group.start_row
            tokens.append(batch.token_ids[rows])
            documents.append(batch.document_ids[rows])
        return PackedRows(
            np.concatenate(tokens),
            np.concatenate(documents),
        )


@dataclass(frozen=True)
class HostTokenBatch:
    token_ids: NDArray[np.int32]

    def __post_init__(self) -> None:
        assert self.token_ids.dtype == np.int32, "token_ids must be int32"
        assert self.token_ids.ndim == 2 and self.token_ids.shape[1] > 0, (
            "token_ids must have shape [batch, sequence] with a nonempty sequence"
        )
        assert np.all(self.token_ids >= 0), "token IDs must be nonnegative"

    def validate_vocabulary(self, vocab_size: int) -> None:
        assert vocab_size > 0, "vocab_size must be positive"
        assert np.all(self.token_ids < vocab_size), (
            f"token IDs outside [0, {vocab_size}); dataset and target vocabularies differ"
        )

    def to_device(self, vocab_size: int) -> LMBatch:
        self.validate_vocabulary(vocab_size)
        return LMBatch(jax.device_put(self.token_ids))


def global_token_batch(
    local: HostTokenBatch, mesh: Mesh, global_batch: int, vocab_size: int
) -> LMBatch:
    local.validate_vocabulary(vocab_size)
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    tokens = jax.make_array_from_process_local_data(
        sharding, local.token_ids, (global_batch, local.token_ids.shape[1])
    )
    return LMBatch(tokens)


def validate_token_shards(shards: tuple[ShardInfo, ...]) -> None:
    for shard in shards:
        assert set(pq.ParquetFile(shard.path).schema_arrow.names) == {"input_ids"}, (
            f"{shard.path}: token-only inputs require exactly input_ids; "
            "this target does not support document-aware inputs"
        )


class TokenShardServer:
    """Serve token-only corpora; document-aware shards are a different input type."""

    def __init__(
        self, schedule: BatchRowSchedule, seq_len: int, process_index: int, process_count: int
    ):
        assert process_count > 0 and 0 <= process_index < process_count
        assert seq_len > 0
        assert schedule.global_batch % process_count == 0
        self.schedule = schedule
        self.seq_len = seq_len
        self.process_index = process_index
        self.process_count = process_count
        self._loaded_row_group: tuple[int, int] | None = None
        self._batch: HostTokenBatch | None = None
        validate_token_shards(schedule.shards)

    @property
    def per_process(self) -> int:
        return self.schedule.global_batch // self.process_count

    def _load_row_group(self, loc: RowGroupSelection) -> HostTokenBatch:
        key = (loc.file_idx, loc.row_group_idx)
        if self._loaded_row_group != key:
            shard = self.schedule.shards[loc.file_idx]
            table = pq.ParquetFile(shard.path).read_row_group(loc.row_group_idx)

            full = HostTokenBatch(read_id_column(table, "input_ids", self.seq_len))
            self._batch = HostTokenBatch(full.token_ids[:, : self.seq_len])
            self._loaded_row_group = key
        assert self._batch is not None
        return self._batch

    def local_batch(self, step: int) -> HostTokenBatch:
        tokens = []
        for selection in _local_batch_rows(
            self.schedule, step, self.process_index, self.process_count
        ):
            group = self.schedule.shards[selection.file_idx].row_groups[selection.row_group_idx]
            batch = self._load_row_group(selection)
            tokens.append(batch.token_ids[selection.rows - group.start_row])
        return HostTokenBatch(np.concatenate(tokens))


class TokenRowSource(Protocol):
    """This process's scheduled rows of token IDs, whatever the corpus format."""

    @property
    def seq_len(self) -> int: ...

    @property
    def per_process(self) -> int: ...

    def local_batch(self, step: int) -> HostTokenBatch: ...


@dataclass(frozen=True)
class DocumentCorpusTokens:
    """A document-aware corpus's token IDs, its document IDs set aside: how a model that
    reads whole rows reads a corpus that records documents."""

    server: ShardServer

    @property
    def seq_len(self) -> int:
        return self.server.seq_len

    @property
    def per_process(self) -> int:
        return self.server.per_process

    def local_batch(self, step: int) -> HostTokenBatch:
        return HostTokenBatch(self.server.local_batch(step).token_ids)


def stores_document_ids(shards: tuple[ShardInfo, ...]) -> bool:
    """Whether the shards pair `input_ids` with `document_ids`."""
    return "document_ids" in pq.ParquetFile(shards[0].path).schema_arrow.names


def token_row_source(
    schedule: BatchRowSchedule, seq_len: int, process_index: int, process_count: int
) -> TokenRowSource:
    """`seq_len` token IDs per scheduled row, from document-aware shards (their document IDs
    set aside) or, to reopen finished runs that trained on them, token-only ones."""
    if stores_document_ids(schedule.shards):
        return DocumentCorpusTokens(ShardServer(schedule, seq_len, process_index, process_count))
    return TokenShardServer(schedule, seq_len, process_index, process_count)
