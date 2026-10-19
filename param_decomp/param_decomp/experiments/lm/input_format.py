"""How an LM target reads its corpus: the row reader it opens and the batch it builds.

The target family owns the choice; training and every offline consumer share it."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
from jax.sharding import Mesh
from param_decomp_formats.dataset_store import DataLayout, read_dataset_meta
from param_decomp_formats.datasets import scan_shards
from param_decomp_formats.packed_rows import PackedRows

from param_decomp.core.sharding import local_data_parallel_size
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.batch_data import (
    ShardServer,
    TokenRowSource,
    global_lm_batch,
    global_token_batch,
    packed_rows_on_device,
    token_row_source,
)
from param_decomp.lm.batch_schedule import BatchRowSchedule, BatchSchedule, PrefixSchedule


class RowReader[TargetIn](Protocol):
    """This process's scheduled rows, as the target's input."""

    @property
    def seq_len(self) -> int:
        """The corpus's row length, as its own metadata records it."""
        ...

    @property
    def per_process(self) -> int: ...

    def device_batch(self, step: int, vocab_size: int) -> TargetIn:
        """This process's rows on its default device."""
        ...

    def global_batch(self, step: int, mesh: Mesh, global_batch: int, vocab_size: int) -> TargetIn:
        """Every process's rows, sharded over the mesh's batch axes."""
        ...


class InputFormat[TargetIn](Protocol):
    """How a target family reads a corpus directory."""

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> RowReader[TargetIn]:
        """`schedule`'s rows of the corpus at `directory`; every scheduled shard lives there."""
        ...


@dataclass(frozen=True)
class DocumentRows:
    """Document-aware shards: packed rows with their document identity."""

    server: ShardServer

    @property
    def seq_len(self) -> int:
        return self.server.seq_len

    @property
    def per_process(self) -> int:
        return self.server.per_process

    def device_batch(self, step: int, vocab_size: int) -> LMBatchWithDocuments:
        return packed_rows_on_device(self.server.local_batch(step), vocab_size)

    def global_batch(
        self, step: int, mesh: Mesh, global_batch: int, vocab_size: int
    ) -> LMBatchWithDocuments:
        return global_lm_batch(self.server.local_batch(step), mesh, global_batch, vocab_size)


@dataclass(frozen=True)
class UnsegmentedTokenRows:
    """Any corpus's token IDs, served as rows that are each one unpadded document."""

    source: TokenRowSource

    def _host_rows(self, step: int) -> PackedRows:
        return PackedRows.from_unsegmented_sequences(self.source.local_batch(step).token_ids)

    @property
    def seq_len(self) -> int:
        return self.source.seq_len

    @property
    def per_process(self) -> int:
        return self.source.per_process

    def device_batch(self, step: int, vocab_size: int) -> LMBatchWithDocuments:
        return packed_rows_on_device(self._host_rows(step), vocab_size)

    def global_batch(
        self, step: int, mesh: Mesh, global_batch: int, vocab_size: int
    ) -> LMBatchWithDocuments:
        return global_lm_batch(self._host_rows(step), mesh, global_batch, vocab_size)


@dataclass(frozen=True)
class TokenRows:
    """Any corpus's token IDs, served as plain token rows with no sequence layout."""

    source: TokenRowSource

    @property
    def seq_len(self) -> int:
        return self.source.seq_len

    @property
    def per_process(self) -> int:
        return self.source.per_process

    def device_batch(self, step: int, vocab_size: int) -> LMBatch:
        return self.source.local_batch(step).to_device(vocab_size)

    def global_batch(self, step: int, mesh: Mesh, global_batch: int, vocab_size: int) -> LMBatch:
        return global_token_batch(self.source.local_batch(step), mesh, global_batch, vocab_size)


def _assert_scheduled_from(directory: Path, schedule: BatchRowSchedule) -> None:
    stray = [shard.path for shard in schedule.shards if shard.path.parent != directory]
    assert not stray, f"schedule reads shards outside {directory}: {stray}"


def _token_rows(
    directory: Path,
    schedule: BatchRowSchedule,
    seq_len: int,
    process_index: int,
    process_count: int,
) -> TokenRowSource:
    _assert_scheduled_from(directory, schedule)
    return token_row_source(schedule, seq_len, process_index, process_count)


@dataclass(frozen=True)
class DocumentInput:
    """Versioned document-aware corpora, read `seq_len` tokens per row. A stored row may
    hold one more token, a pretraining label, which is dropped."""

    seq_len: int

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> DocumentRows:
        _assert_scheduled_from(directory, schedule)
        read_dataset_meta(directory)  # refuses corpora without versioned document metadata
        return DocumentRows(ShardServer(schedule, self.seq_len, process_index, process_count))


@dataclass(frozen=True)
class UnsegmentedTokenInput:
    """Any corpus read as whole rows, each one unpadded document of `seq_len` tokens."""

    seq_len: int

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> UnsegmentedTokenRows:
        return UnsegmentedTokenRows(
            _token_rows(directory, schedule, self.seq_len, process_index, process_count)
        )


@dataclass(frozen=True)
class TokenInput:
    """Any corpus read as token rows of `seq_len` tokens, with no sequence layout."""

    seq_len: int

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> TokenRows:
        return TokenRows(
            _token_rows(directory, schedule, self.seq_len, process_index, process_count)
        )


def transformer_input_format(
    layout: DataLayout, seq_len: int
) -> DocumentInput | UnsegmentedTokenInput:
    """The rows a shared-transformer target reads under its run's declared layout."""
    match layout:
        case "documents":
            return DocumentInput(seq_len)
        case "unsegmented_tokens":
            return UnsegmentedTokenInput(seq_len)


def qwen36_moe_input_format(layout: DataLayout, seq_len: int) -> TokenInput:
    """The rows the Qwen3.6 MoE target reads: token rows, with no sequence layout."""
    match layout:
        case "unsegmented_tokens":
            return TokenInput(seq_len)
        case "documents":
            raise AssertionError("the Qwen3.6 MoE target does not read document-aware rows")


def input_sampler[TargetIn](
    input_format: InputFormat[TargetIn],
    directory: Path,
    batch_size: int,
    seed: int,
    mesh: Mesh,
    vocab_size: int,
) -> Callable[[int], TargetIn]:
    """The training batch for each step, drawn across every process."""
    return _sampler(
        input_format,
        directory,
        BatchSchedule(scan_shards(directory), batch_size, seed),
        mesh,
        vocab_size,
    )


def eval_sampler[TargetIn](
    input_format: InputFormat[TargetIn],
    directory: Path,
    batch_size: int,
    n_batches: int,
    mesh: Mesh,
    vocab_size: int,
) -> Callable[[int], TargetIn]:
    """Batch `j < n_batches` of the fixed eval set: the eval dataset's first rows, drawn
    across every process."""
    return _sampler(
        input_format,
        directory,
        PrefixSchedule(scan_shards(directory), batch_size, n_batches),
        mesh,
        vocab_size,
    )


def _sampler[TargetIn](
    input_format: InputFormat[TargetIn],
    directory: Path,
    schedule: BatchRowSchedule,
    mesh: Mesh,
    vocab_size: int,
) -> Callable[[int], TargetIn]:
    reader = input_format.reader(directory, schedule, jax.process_index(), jax.process_count())
    assert reader.per_process % local_data_parallel_size(mesh) == 0
    return lambda step: reader.global_batch(step, mesh, schedule.global_batch, vocab_size)
