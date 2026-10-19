"""Each input format validates its own corpus representation, and a run's declared layout
selects it."""

from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax.sharding import AxisType, Mesh
from param_decomp_formats.dataset_store import DatasetIdentity, DatasetMeta
from pydantic import ValidationError

from param_decomp.experiments.lm.input_format import (
    DocumentInput,
    TokenInput,
    input_sampler,
    qwen36_moe_input_format,
    transformer_input_format,
)


def _write_corpus(directory: Path, metadata: DatasetIdentity, documents: bool) -> None:
    (directory / "meta.json").write_text(metadata.model_dump_json())
    columns = {"input_ids": pa.array([[1, 2, 3, 4]] * 4, type=pa.list_(pa.int32()))}
    if documents:
        columns["document_ids"] = pa.array([[0, 0, 1, 1]] * 4, type=pa.list_(pa.int32()))
    pq.write_table(pa.table(columns), directory / "train-00000-of-00001.parquet")


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )


_VERSIONED = DatasetMeta(
    seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"
)


def test_token_input_reads_legacy_metadata_without_rewriting_it(tmp_path: Path) -> None:
    _write_corpus(tmp_path, DatasetIdentity(seq_len=4, tokenizer_name="test"), documents=False)
    before = (tmp_path / "meta.json").read_bytes()
    mesh = _mesh()
    with jax.set_mesh(mesh):
        sample = input_sampler(TokenInput(4), tmp_path, 2, 3, mesh, 10)
        np.testing.assert_array_equal(sample(0).token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(sample(5).token_ids, [[1, 2, 3, 4]] * 2)
    assert (tmp_path / "meta.json").read_bytes() == before


def test_token_input_reads_a_document_aware_corpus_s_tokens(tmp_path: Path) -> None:
    _write_corpus(tmp_path, _VERSIONED, documents=True)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(TokenInput(4), tmp_path, 2, 3, mesh, 10)(0)
        np.testing.assert_array_equal(batch.token_ids, [[1, 2, 3, 4]] * 2)


def test_document_input_rejects_unversioned_metadata(tmp_path: Path) -> None:
    _write_corpus(tmp_path, DatasetIdentity(seq_len=4, tokenizer_name="test"), documents=True)
    with pytest.raises(ValidationError, match="format_version"):
        input_sampler(DocumentInput(4), tmp_path, 2, 3, _mesh(), 10)


def test_document_input_rejects_missing_document_columns(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"),
        documents=False,
    )
    sample = input_sampler(DocumentInput(4), tmp_path, 2, 3, _mesh(), 10)
    with pytest.raises(AssertionError, match="shards require input_ids and document_ids"):
        sample(0)


def test_document_input_drops_a_stored_pretraining_label(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(
        DatasetMeta(
            seq_len=5, tokenizer_name="test", format_version=2, preprocessing_name="test"
        ).model_dump_json()
    )
    columns = {
        "input_ids": pa.array([[1, 2, 3, 4, 5]] * 4, type=pa.list_(pa.int32())),
        "document_ids": pa.array([[0, 0, 1, 1, 1]] * 4, type=pa.list_(pa.int32())),
    }
    pq.write_table(pa.table(columns), tmp_path / "train-00000-of-00001.parquet")
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(DocumentInput(4), tmp_path, 2, 3, mesh, 10)(0)
        np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 1, 1]] * 2)


def test_document_input_preserves_paired_inputs(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        DatasetMeta(seq_len=4, tokenizer_name="test", format_version=2, preprocessing_name="test"),
        documents=True,
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(DocumentInput(4), tmp_path, 2, 3, mesh, 10)(0)
        np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 1, 1]] * 2)


@pytest.mark.parametrize("documents", [False, True])
def test_document_layout_requires_stored_document_ids(tmp_path: Path, documents: bool) -> None:
    _write_corpus(tmp_path, _VERSIONED, documents=documents)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        sample = input_sampler(transformer_input_format("documents", 4), tmp_path, 2, 3, mesh, 10)
        if documents:
            batch = sample(0)
            np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
            np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 1, 1]] * 2)
        else:
            with pytest.raises(AssertionError, match="shards require input_ids and document_ids"):
                sample(0)


@pytest.mark.parametrize("versioned", [False, True])
def test_unsegmented_layout_preserves_token_rows(tmp_path: Path, versioned: bool) -> None:
    metadata = _VERSIONED if versioned else DatasetIdentity(seq_len=4, tokenizer_name="test")
    _write_corpus(tmp_path, metadata, documents=False)
    before = (tmp_path / "meta.json").read_bytes()
    mesh = _mesh()
    layout_format = transformer_input_format("unsegmented_tokens", 4)
    with jax.set_mesh(mesh):
        sample = input_sampler(layout_format, tmp_path, 2, 3, mesh, 10)
        for step in (0, 5):
            batch = sample(step)
            np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
            np.testing.assert_array_equal(batch.sequence.document_ids, np.zeros((2, 4)))
            np.testing.assert_array_equal(batch.sequence.position_ids(), [[0, 1, 2, 3]] * 2)
    assert (tmp_path / "meta.json").read_bytes() == before


def test_unsegmented_layout_reads_stored_label_rows_at_the_declared_width(tmp_path: Path) -> None:
    """Whole rows of a pretraining corpus (stored width seq_len + 1) are read at seq_len,
    the label dropped and the document IDs set aside."""
    (tmp_path / "meta.json").write_text(
        DatasetMeta(
            seq_len=5, tokenizer_name="test", format_version=2, preprocessing_name="test"
        ).model_dump_json()
    )
    columns = {
        "input_ids": pa.array([[1, 2, 3, 4, 5]] * 4, type=pa.list_(pa.int32())),
        "document_ids": pa.array([[0, 0, 1, 1, 1]] * 4, type=pa.list_(pa.int32())),
    }
    pq.write_table(pa.table(columns), tmp_path / "train-00000-of-00001.parquet")
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(
            transformer_input_format("unsegmented_tokens", 4), tmp_path, 2, 3, mesh, 10
        )(0)
        np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(batch.sequence.document_ids, np.zeros((2, 4)))


def test_unsegmented_layout_sets_a_document_aware_corpus_s_ids_aside(tmp_path: Path) -> None:
    _write_corpus(tmp_path, _VERSIONED, documents=True)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        batch = input_sampler(
            transformer_input_format("unsegmented_tokens", 4), tmp_path, 2, 3, mesh, 10
        )(0)
        np.testing.assert_array_equal(batch.batch.token_ids, [[1, 2, 3, 4]] * 2)
        np.testing.assert_array_equal(batch.sequence.document_ids, np.zeros((2, 4)))


def test_qwen36_moe_refuses_document_layout() -> None:
    assert isinstance(qwen36_moe_input_format("unsegmented_tokens", 4), TokenInput)
    with pytest.raises(AssertionError, match="does not read document-aware rows"):
        qwen36_moe_input_format("documents", 4)
