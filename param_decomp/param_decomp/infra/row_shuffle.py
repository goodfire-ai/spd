"""Write a seeded, uniformly row-shuffled copy of a dataset: a document-aware one (token and
document IDs per row) or a historical token-only one. Tokenizing ends with it
(`experiments.lm.llama3.prestage_tokenized finish`), so every published dataset's rows are
shuffled.

The training loader (`BatchSchedule`) cuts each batch from consecutive rows of one or two
Parquet row groups. A corpus stored in source order therefore yields batches of neighbouring
rows from one source file. This tool permutes whole rows across the corpus; a row's columns
are never edited, so document identity stays consistent within each row.

Two passes through a staging directory on the output's filesystem:

- scatter: each source shard's rows draw seeded random buckets and are written as one staging
  file whose row groups are its nonempty buckets, in bucket order.
- gather: each bucket's rows from every staging file, permuted, become one output shard.

Before publishing, every output shard is reread and its rows' order-independent fingerprint
must equal the source's. `row_shuffle.json` records the source, the digests of its metadata
and any manifest, the seed and the fingerprint.
"""

import hashlib
import json
import math
import multiprocessing
import os
import resource
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from param_decomp_formats.dataset_store import (
    DATASET_META_FILENAME,
    ROW_SHUFFLE_FILENAME,
    DatasetIdentity,
    DatasetMeta,
    write_dataset_meta,
)

from param_decomp.core.log import logger

SOURCE_MANIFEST_FILENAMES = ("dataset_manifest.json", "manifest.json")
DOCUMENT_COLUMNS = ("input_ids", "document_ids")
TOKEN_COLUMNS = ("input_ids",)
OUTPUT_ROW_GROUP_ROWS = 8192
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class RowsFingerprint:
    """An order-independent summary of a multiset of rows: its size and the wrapped sums of
    each row's 64-bit hash and squared hash. Equal multisets give equal fingerprints."""

    rows: int
    hash_sum: int
    hash_square_sum: int

    def __add__(self, other: "RowsFingerprint") -> "RowsFingerprint":
        return RowsFingerprint(
            self.rows + other.rows,
            (self.hash_sum + other.hash_sum) & _MASK64,
            (self.hash_square_sum + other.hash_square_sum) & _MASK64,
        )


EMPTY_FINGERPRINT = RowsFingerprint(0, 0, 0)


def fingerprint(*columns: NDArray[np.int32]) -> RowsFingerprint:
    """Hash each row as a fixed random odd-weighted sum of its IDs in every column (each
    `[rows, width]`): its tokens, and its document IDs where the dataset has them."""
    shape = columns[0].shape
    assert len(shape) == 2 and all(column.shape == shape for column in columns), [
        column.shape for column in columns
    ]
    weights = np.random.default_rng(0x5EED).integers(
        0, 1 << 63, size=(len(columns), shape[1]), dtype=np.uint64
    ) * np.uint64(2) + np.uint64(1)
    with np.errstate(over="ignore"):
        hashes = sum(
            (column.astype(np.uint64) @ w for column, w in zip(columns, weights, strict=True)),
            np.zeros(shape[0], np.uint64),
        )
        squares = hashes * hashes
    return RowsFingerprint(shape[0], int(hashes.sum()), int(squares.sum()))


def table_fingerprint(table: object) -> RowsFingerprint:
    rows = pa.table(table)
    if len(rows) == 0:
        return EMPTY_FINGERPRINT
    return fingerprint(
        *(
            rows.column(name).combine_chunks().flatten().to_numpy().reshape(len(rows), -1)
            for name in rows.column_names
        )
    )


@dataclass(frozen=True)
class DocumentRows:
    """A document-aware dataset: versioned metadata, rows of token and document IDs."""

    meta: DatasetMeta


@dataclass(frozen=True)
class TokenRows:
    """A historical token-only dataset: identity-only metadata, rows of token IDs."""

    identity: DatasetIdentity


def read_source_rows(source_dir: Path) -> DocumentRows | TokenRows:
    """Which kind of dataset `source_dir` is, by its metadata: versioned metadata names a
    document-aware dataset; identity alone, a historical token-only one."""
    raw = json.loads((source_dir / DATASET_META_FILENAME).read_text())
    if "format_version" in raw:
        return DocumentRows(DatasetMeta.model_validate(raw))
    return TokenRows(DatasetIdentity.model_validate(raw))


@dataclass(frozen=True)
class _Plan:
    source_files: tuple[Path, ...]
    staging: Path
    out: Path
    seed: int
    buckets: int
    schema: pa.Schema

    def staging_file(self, source_idx: int) -> Path:
        return self.staging / f"scatter_{source_idx:05d}.parquet"

    def output_file(self, bucket: int) -> Path:
        return self.out / f"shard_{bucket:05d}.parquet"


def _scatter(plan: _Plan, source_idx: int) -> tuple[RowsFingerprint, NDArray[np.int64]]:
    """Stage one source shard by bucket; returns its fingerprint and per-bucket row counts."""
    table = pq.read_table(plan.source_files[source_idx])
    assert table.schema.equals(plan.schema), plan.source_files[source_idx]
    rng = np.random.default_rng((plan.seed, 0, source_idx))
    buckets = rng.integers(0, plan.buckets, size=len(table))
    counts = np.bincount(buckets, minlength=plan.buckets)
    ordered = table.take(np.argsort(buckets, kind="stable"))
    with pq.ParquetWriter(
        plan.staging_file(source_idx), plan.schema, compression="zstd", compression_level=1
    ) as writer:
        start = 0
        for count in counts[counts > 0]:
            writer.write_table(ordered.slice(start, int(count)), row_group_size=int(count))
            start += int(count)
    return table_fingerprint(table), counts


def _gather(
    plan: _Plan, first_bucket: int, counts: NDArray[np.int64], preceding: NDArray[np.int64]
) -> RowsFingerprint:
    """Write the output shards for buckets `first_bucket ...`, one per nonempty column of
    `counts[source, bucket - first_bucket]` (the rows each staging file holds for it); returns
    the reread shards' fingerprint.

    A staging file's row groups are its nonempty buckets in order, `preceding[source]` of them
    before `first_bucket`. Each staging file is opened once for the whole range."""
    files = [pq.ParquetFile(plan.staging_file(i)) for i in range(len(plan.source_files))]
    total = EMPTY_FINGERPRINT
    for offset in range(counts.shape[1]):
        bucket = first_bucket + offset
        parts = []
        for source_idx, parquet in enumerate(files):
            count = int(counts[source_idx, offset])
            if count == 0:
                continue
            row_group = int(preceding[source_idx] + np.count_nonzero(counts[source_idx, :offset]))
            part = parquet.read_row_group(row_group)
            assert len(part) == count, (source_idx, bucket, len(part), count)
            parts.append(part)
        if not parts:
            continue
        rows = pa.concat_tables(parts)
        rng = np.random.default_rng((plan.seed, 1, bucket))
        pq.write_table(
            rows.take(rng.permutation(len(rows))),
            plan.output_file(bucket),
            row_group_size=OUTPUT_ROW_GROUP_ROWS,
            compression="zstd",
        )
        total = total + table_fingerprint(pq.read_table(plan.output_file(bucket)))
    return total


def _allow_open_files(needed: int) -> None:
    """Raise this process's open-file limit, which the pool's workers inherit, to `needed`:
    each gather task holds every staging file open, one per source shard."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert hard == resource.RLIM_INFINITY or needed <= hard, (
        f"gathering needs {needed} open files; the hard limit is {hard}"
    )
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shuffle_rows(
    source_dir: Path,
    out_dir: Path,
    staging_dir: Path,
    seed: int,
    rows_per_shard: int,
    buckets_per_task: int,
    workers: int,
    producer_revision: str,
) -> RowsFingerprint:
    """Publish `out_dir` as a row-shuffled copy of `source_dir`; returns its fingerprint."""
    assert rows_per_shard > 0 and buckets_per_task > 0 and workers > 0, (
        rows_per_shard,
        buckets_per_task,
        workers,
    )
    source = read_source_rows(source_dir)
    match source:
        case DocumentRows():
            columns = DOCUMENT_COLUMNS
        case TokenRows():
            columns = TOKEN_COLUMNS
    source_digests = {
        name: _sha256(source_dir / name)
        for name in (DATASET_META_FILENAME, *SOURCE_MANIFEST_FILENAMES)
        if (source_dir / name).exists()
    }
    source_files = tuple(sorted(source_dir.glob("*.parquet")))
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in source_files)
    assert total_rows > 0, f"no rows in {source_dir}"
    assert not out_dir.exists(), f"refusing to replace {out_dir}"
    staging_dir.mkdir(parents=True)
    assert os.stat(staging_dir).st_dev == os.stat(out_dir.parent).st_dev, (
        "staging must share the output's filesystem so publishing is a rename"
    )
    plan = _Plan(
        source_files=source_files,
        staging=staging_dir,
        out=staging_dir / "out",
        seed=seed,
        buckets=math.ceil(total_rows / rows_per_shard),
        schema=pq.read_schema(source_files[0]),
    )
    assert tuple(plan.schema.names) == columns, plan.schema
    plan.out.mkdir()
    logger.info(f"{total_rows} rows from {len(source_files)} shards -> {plan.buckets} shards")

    _allow_open_files(len(source_files) + 64)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        scattered = list(pool.map(_scatter, [plan] * len(source_files), range(len(source_files))))
        source_fingerprint = sum((fp for fp, _ in scattered), EMPTY_FINGERPRINT)
        counts = np.stack([bucket_counts for _, bucket_counts in scattered])
        logger.info(f"scattered {source_fingerprint.rows} rows")
        starts = range(0, plan.buckets, buckets_per_task)
        output = sum(
            pool.map(
                _gather,
                [plan] * len(starts),
                starts,
                [counts[:, b : b + buckets_per_task] for b in starts],
                [np.count_nonzero(counts[:, :b], axis=1) for b in starts],
            ),
            EMPTY_FINGERPRINT,
        )
    assert output == source_fingerprint, (output, source_fingerprint)

    match source:
        case DocumentRows(meta=meta):
            write_dataset_meta(
                plan.out,
                meta.model_copy(
                    update={
                        "preprocessing_name": f"{meta.preprocessing_name}+row-shuffle-seed{seed}"
                    }
                ),
            )
        case TokenRows(identity=identity):
            (plan.out / DATASET_META_FILENAME).write_text(identity.model_dump_json() + "\n")
    (plan.out / ROW_SHUFFLE_FILENAME).write_text(
        json.dumps(
            {
                "producer": "param_decomp.infra.row_shuffle",
                "producer_revision": producer_revision,
                "source_dir": str(source_dir),
                "source_sha256": source_digests,
                "seed": seed,
                "shards": len(list(plan.out.glob("*.parquet"))),
                "output_row_group_rows": OUTPUT_ROW_GROUP_ROWS,
                "fingerprint": asdict(output),
            },
            indent=2,
        )
        + "\n"
    )
    plan.out.rename(out_dir)
    shutil.rmtree(staging_dir)
    logger.info(f"published {out_dir}: {output}")
    return output
