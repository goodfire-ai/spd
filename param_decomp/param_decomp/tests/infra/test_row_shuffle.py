import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray
from param_decomp_formats.dataset_store import (
    ROW_SHUFFLE_FILENAME,
    DatasetIdentity,
    DatasetMeta,
    assert_row_shuffled,
    read_dataset_identity,
    read_dataset_meta,
    write_dataset_meta,
)
from param_decomp_formats.datasets import scan_shards

from param_decomp.infra.row_shuffle import (
    SOURCE_MANIFEST_FILENAMES,
    fingerprint,
    shuffle_rows,
)
from param_decomp.lm.batch_data import ShardServer
from param_decomp.lm.batch_schedule import BatchSchedule

WIDTH = 9


def _table(tokens: NDArray[np.int32], documents: NDArray[np.int32]) -> object:
    column_type = pa.list_(pa.int32(), WIDTH)
    return pa.table(
        {
            "input_ids": pa.FixedSizeListArray.from_arrays(pa.array(tokens.ravel()), WIDTH).cast(
                column_type
            ),
            "document_ids": pa.FixedSizeListArray.from_arrays(
                pa.array(documents.ravel()), WIDTH
            ).cast(column_type),
        }
    )


def _source_order_dataset(root: Path, shard_rows: tuple[int, ...]) -> NDArray[np.int32]:
    """Shards whose rows are numbered consecutively in token 0, with two documents per row;
    returns every row's tokens in source order."""
    root.mkdir()
    write_dataset_meta(
        root,
        DatasetMeta(
            format_version=2, seq_len=WIDTH, tokenizer_name="synthetic", preprocessing_name="toy"
        ),
    )
    (root / SOURCE_MANIFEST_FILENAMES[0]).write_text("{}\n")
    rows = []
    first = 0
    for shard, n in enumerate(shard_rows):
        tokens = (np.arange(first, first + n)[:, None] * 100 + np.arange(WIDTH)).astype(np.int32)
        documents = (np.arange(n)[:, None] * 2 + (np.arange(WIDTH) >= WIDTH // 2)).astype(np.int32)
        pq.write_table(
            _table(tokens, documents), root / f"src_{shard:05d}.parquet", row_group_size=7
        )
        rows.append(tokens)
        first += n
    return np.concatenate(rows)


def _read_tokens(directory: Path) -> NDArray[np.int32]:
    tables = [pq.read_table(path) for path in sorted(directory.glob("*.parquet"))]
    return np.concatenate(
        [
            t.column("input_ids").combine_chunks().flatten().to_numpy().reshape(len(t), -1)
            for t in tables
        ]
    )


def test_shuffle_preserves_rows_and_mixes_sources(tmp_path: Path):
    source_dir = tmp_path / "source"
    shard_rows = (40, 25, 3, 52)
    source = _source_order_dataset(source_dir, shard_rows)
    out_dir = tmp_path / "shuffled"
    published = shuffle_rows(
        source_dir,
        out_dir,
        tmp_path / "staging",
        seed=7,
        rows_per_shard=30,
        buckets_per_task=2,
        workers=2,
        producer_revision="test",
    )

    shuffled = _read_tokens(out_dir)
    assert len(list(out_dir.glob("*.parquet"))) == 4
    assert published.rows == len(source) == len(shuffled)
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, source))
    assert not np.array_equal(shuffled, source)
    first_shard = shuffled[: pq.ParquetFile(sorted(out_dir.glob("*.parquet"))[0]).metadata.num_rows]
    origin = np.searchsorted(np.cumsum(shard_rows), first_shard[:, 0] // 100, side="right")
    assert len(set(origin.tolist())) > 1
    assert not (tmp_path / "staging").exists()

    meta = read_dataset_meta(out_dir)
    assert meta.seq_len == WIDTH and meta.preprocessing_name == "toy+row-shuffle-seed7"
    assert_row_shuffled(out_dir)
    with pytest.raises(AssertionError, match="records no row shuffle"):
        assert_row_shuffled(source_dir)
    record = json.loads((out_dir / ROW_SHUFFLE_FILENAME).read_text())
    assert record["seed"] == 7 and record["fingerprint"]["rows"] == len(source)
    assert record["source_sha256"].keys() == {"meta.json", SOURCE_MANIFEST_FILENAMES[0]}

    shards = scan_shards(out_dir)
    server = ShardServer(BatchSchedule(shards, 8, 0), WIDTH, 0, 1)
    batch = server.local_batch(0)
    assert batch.token_ids.shape == batch.document_ids.shape == (8, WIDTH)


def test_shuffle_is_deterministic_in_its_seed(tmp_path: Path):
    source_dir = tmp_path / "source"
    _source_order_dataset(source_dir, (30, 30))
    outputs = []
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):
        shuffle_rows(
            source_dir, tmp_path / name, tmp_path / f"staging_{name}", seed, 20, 1, 1, "test"
        )
        outputs.append(_read_tokens(tmp_path / name))
    assert np.array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0], outputs[2])


def test_fingerprint_ignores_order_and_detects_any_changed_id():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50_000, size=(16, WIDTH), dtype=np.int32)
    documents = np.zeros_like(tokens)
    base = fingerprint(tokens, documents)
    assert fingerprint(tokens[::-1], documents) == base
    changed = documents.copy()
    changed[3, 5] = 1
    assert fingerprint(tokens, changed) != base


def test_a_token_only_dataset_keeps_its_identity_metadata(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    identity = DatasetIdentity(seq_len=WIDTH, tokenizer_name="synthetic")
    (source_dir / "meta.json").write_text(identity.model_dump_json())
    tokens = (np.arange(30)[:, None] * 100 + np.arange(WIDTH)).astype(np.int32)
    column_type = pa.list_(pa.int32(), WIDTH)
    table = pa.table(
        {
            "input_ids": pa.FixedSizeListArray.from_arrays(pa.array(tokens.ravel()), WIDTH).cast(
                column_type
            )
        }
    )
    pq.write_table(table, source_dir / "shard_00000.parquet", row_group_size=7)
    out_dir = tmp_path / "shuffled"
    published = shuffle_rows(source_dir, out_dir, tmp_path / "staging", 3, 10, 1, 1, "test")
    shuffled = _read_tokens(out_dir)
    assert published == fingerprint(tokens)
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, tokens))
    assert not np.array_equal(shuffled, tokens)
    assert read_dataset_identity(out_dir) == identity
    assert_row_shuffled(out_dir)


def test_empty_source_shards_and_empty_buckets_are_skipped(tmp_path: Path):
    source_dir = tmp_path / "source"
    source = _source_order_dataset(source_dir, (0, 10, 2))
    out_dir = tmp_path / "shuffled"
    published = shuffle_rows(source_dir, out_dir, tmp_path / "staging", 0, 1, 1, 1, "test")
    shuffled = _read_tokens(out_dir)
    assert published.rows == len(source) == 12
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, source))
    record = json.loads((out_dir / ROW_SHUFFLE_FILENAME).read_text())
    assert record["shards"] == len(list(out_dir.glob("*.parquet"))) <= 12


def test_shuffle_refuses_an_empty_dataset_before_staging(tmp_path: Path):
    source_dir = tmp_path / "source"
    _source_order_dataset(source_dir, (0,))
    with pytest.raises(AssertionError, match="no rows"):
        shuffle_rows(source_dir, tmp_path / "out", tmp_path / "staging", 0, 5, 1, 1, "test")
    assert not (tmp_path / "staging").exists()


def test_shuffle_refuses_an_existing_output(tmp_path: Path):
    source_dir = tmp_path / "source"
    _source_order_dataset(source_dir, (10,))
    (tmp_path / "out").mkdir()
    with pytest.raises(AssertionError, match="refusing to replace"):
        shuffle_rows(source_dir, tmp_path / "out", tmp_path / "staging", 0, 5, 1, 1, "test")
