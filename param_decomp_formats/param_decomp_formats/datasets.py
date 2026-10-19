"""Rows as a tokenized dataset stores them: Parquet shards of packed rows, each a
row's token ids and its tokens' document labels, read on the host exactly as training
reads them."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from param_decomp_formats.packed_rows import PackedRows


@dataclass(frozen=True)
class RowGroupInfo:
    start_row: int
    n_rows: int


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    row_groups: tuple[RowGroupInfo, ...]

    @property
    def n_rows(self) -> int:
        return sum(group.n_rows for group in self.row_groups)


def scan_shards(data_dir: Path) -> tuple[ShardInfo, ...]:
    files = sorted(data_dir.glob("*.parquet"))
    assert files, f"no *.parquet under {data_dir}"

    shards = []
    for path in files:
        metadata = pq.ParquetFile(path).metadata
        row_groups = []
        start_row = 0
        for row_group_idx in range(metadata.num_row_groups):
            n_rows = metadata.row_group(row_group_idx).num_rows
            row_groups.append(RowGroupInfo(start_row=start_row, n_rows=n_rows))
            start_row += n_rows
        assert start_row == metadata.num_rows
        shards.append(ShardInfo(path=path, row_groups=tuple(row_groups)))
    return tuple(shards)


def read_id_column(table: object, name: str, seq_len: int) -> NDArray[np.int32]:
    """The int32 list column `name` of a row group's `table`, as `[rows, width]`, every
    row `seq_len` or `seq_len + 1` wide."""
    """The int32 list column `name` of a row group's `table`, as `[rows, width]`, every
    row `seq_len` or `seq_len + 1` wide."""
    ids = pa.table(table).column(name).combine_chunks()
    assert (
        pa.types.is_list(ids.type)
        or pa.types.is_large_list(ids.type)
        or pa.types.is_fixed_size_list(ids.type)
    ) and ids.type.value_type == pa.int32(), f"{name} must be an int32 list"
    assert ids.null_count == 0, f"{name} has null rows"
    lengths = pc.call_function("list_value_length", [ids]).to_numpy(zero_copy_only=False)
    assert len(lengths) > 0 and np.all(lengths == lengths[0]), (
        f"{name} rows must have equal lengths"
    )
    assert lengths[0] in (seq_len, seq_len + 1), (
        f"{name} rows have seq {lengths[0]}, config says {seq_len}"
    )
    flat = ids.flatten()
    assert flat.null_count == 0, f"{name} has null IDs"
    return flat.to_numpy(zero_copy_only=False).reshape(len(lengths), -1)


type OpenShard = Callable[[Path], pq.ParquetFile]
"""How a reader opens a shard by its `ShardInfo.path`: a local file, by default."""


def read_row_group(
    shard: ShardInfo, row_group_idx: int, seq_len: int, open_shard: OpenShard = pq.ParquetFile
) -> PackedRows:
    """One row group's packed rows, each cut to `seq_len`, as training reads them."""
    table = open_shard(shard.path).read_row_group(row_group_idx)
    assert set(table.column_names) == {"input_ids", "document_ids"}, (
        f"{shard.path}: shards require input_ids and document_ids; tokenize older datasets again"
    )
    full = PackedRows(
        read_id_column(table, "input_ids", seq_len), read_id_column(table, "document_ids", seq_len)
    )
    return PackedRows(full.token_ids[:, :seq_len], full.document_ids[:, :seq_len])


def read_shard_rows(
    shard: ShardInfo, rows: Sequence[int], seq_len: int, open_shard: OpenShard = pq.ParquetFile
) -> PackedRows:
    """Rows `rows` of one shard, opened by `open_shard`, in that order, as training reads
    them."""
    assert all(0 <= row < shard.n_rows for row in rows), (rows, shard.n_rows)
    starts = np.array([group.start_row for group in shard.row_groups])
    groups = np.searchsorted(starts, rows, side="right") - 1
    read = {
        int(group): read_row_group(shard, int(group), seq_len, open_shard) for group in set(groups)
    }
    local_rows = np.asarray(rows) - starts[groups]
    placed = list(zip(groups.tolist(), local_rows.tolist(), strict=True))
    return PackedRows(
        np.stack([read[group].token_ids[local] for group, local in placed]),
        np.stack([read[group].document_ids[local] for group, local in placed]),
    )
