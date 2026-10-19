"""The `DatasetRef` schema: store names resolve under `data_root`, ad-hoc dirs are
absolute, and datasets self-describe via `meta.json`."""

import json
from pathlib import Path

import pytest
from param_decomp_formats.dataset_store import (
    DatasetDir,
    DatasetIdentity,
    DatasetMeta,
    NamedDataset,
    assert_records_documents,
    dataset_dir,
    read_dataset_identity,
    read_dataset_meta,
    write_dataset_meta,
)
from pydantic import ValidationError


def test_store_name_resolves_under_data_root() -> None:
    assert dataset_dir(Path("/project"), "pile") == Path("/project/datasets/pile")


def test_dataset_names_are_flat() -> None:
    with pytest.raises(ValidationError, match="flat store names"):
        NamedDataset.model_validate({"kind": "name", "name": "datasets/pile"})


def test_ad_hoc_dirs_are_absolute() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        DatasetDir.model_validate({"kind": "dir", "dir": "relative/shards"})


def test_dataset_meta_round_trips(tmp_path: Path) -> None:
    meta = DatasetMeta(
        format_version=2,
        preprocessing_name="fixture",
        seq_len=512,
        tokenizer_name="EleutherAI/gpt-neox-20b",
    )
    write_dataset_meta(tmp_path, meta)
    assert read_dataset_meta(tmp_path) == meta


def test_dataset_meta_records_its_end_of_document_token(tmp_path: Path) -> None:
    meta = DatasetMeta(
        format_version=2,
        preprocessing_name="fixture",
        seq_len=512,
        tokenizer_name="Qwen/Qwen3-1.7B",
        end_of_document_token_id=151643,
    )
    write_dataset_meta(tmp_path, meta)
    assert read_dataset_meta(tmp_path).end_of_document_token_id == 151643


def test_dataset_meta_without_end_of_document_token_still_reads(tmp_path: Path) -> None:
    """Datasets staged before the field was recorded read as `None`."""
    fields = {"format_version": 2, "preprocessing_name": "f", "seq_len": 8, "tokenizer_name": "t"}
    (tmp_path / "meta.json").write_text(json.dumps(fields))
    assert read_dataset_meta(tmp_path).end_of_document_token_id is None


def test_dataset_meta_missing_refuses(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="self-describing"):
        read_dataset_meta(tmp_path)


def test_new_runs_refuse_token_only_datasets(tmp_path: Path) -> None:
    token_only, document_aware = tmp_path / "token_only", tmp_path / "document_aware"
    token_only.mkdir()
    document_aware.mkdir()
    (token_only / "meta.json").write_text(json.dumps({"seq_len": 512, "tokenizer_name": "t"}))
    with pytest.raises(AssertionError, match="token-only dataset with no document IDs"):
        assert_records_documents(token_only)
    meta = DatasetMeta(format_version=2, preprocessing_name="p", seq_len=513, tokenizer_name="t")
    write_dataset_meta(document_aware, meta)
    assert assert_records_documents(document_aware) == meta


def test_dataset_meta_requires_preprocessing_provenance() -> None:
    with pytest.raises(ValidationError, match="preprocessing_name"):
        DatasetMeta.model_validate({"format_version": 2, "seq_len": 512, "tokenizer_name": "test"})


def test_dataset_meta_rejects_empty_preprocessing_name() -> None:
    with pytest.raises(ValidationError, match="preprocessing_name"):
        DatasetMeta(format_version=2, preprocessing_name="", seq_len=512, tokenizer_name="test")


@pytest.mark.parametrize(
    "storage_fields",
    [
        {},
        {"format_version": 1},
        {"format_version": 2, "preprocessing_name": "fixture"},
        {"format_version": 999, "future_storage_field": "opaque"},
    ],
)
def test_identity_reads_no_shards_and_preserves_metadata(
    tmp_path: Path, storage_fields: dict[str, object]
) -> None:
    path = tmp_path / "meta.json"
    contents = json.dumps({"seq_len": 512, "tokenizer_name": "test", **storage_fields})
    path.write_text(contents)
    assert read_dataset_identity(tmp_path) == DatasetIdentity(seq_len=512, tokenizer_name="test")
    assert path.read_text() == contents
    assert list(tmp_path.iterdir()) == [path]
    if storage_fields.get("format_version") == 2:
        assert read_dataset_meta(tmp_path).format_version == 2
    else:
        with pytest.raises(ValidationError):
            read_dataset_meta(tmp_path)


@pytest.mark.parametrize(
    "identity",
    [
        {"seq_len": 0, "tokenizer_name": "test"},
        {"seq_len": 512, "tokenizer_name": ""},
        {"seq_len": "invalid", "tokenizer_name": "test"},
    ],
)
def test_identity_validates_its_facts(tmp_path: Path, identity: dict[str, object]) -> None:
    (tmp_path / "meta.json").write_text(json.dumps(identity))
    with pytest.raises(ValidationError):
        read_dataset_identity(tmp_path)


@pytest.mark.parametrize("missing", ["seq_len", "tokenizer_name"])
def test_identity_requires_both_facts(tmp_path: Path, missing: str) -> None:
    raw = {"seq_len": 512, "tokenizer_name": "test"}
    del raw[missing]
    (tmp_path / "meta.json").write_text(json.dumps(raw))
    with pytest.raises(KeyError, match=missing):
        read_dataset_identity(tmp_path)


def test_identity_requires_metadata_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dataset_identity(tmp_path)
