"""The dataset-store artifact contract: layout, the reference a config names shards by,
and the self-describing `meta.json`.

A store dataset is a directory of pre-tokenized `*.parquet` shards plus a `meta.json`
carrying the dataset's own facts. A dataset PD reads must also be row-shuffled, recorded by
`row_shuffle.json` (`assert_row_shuffled`). Composition roots and consumers read the meta here
and thread the values into the core loader as explicit parameters
(`ShardServer(seq_len=...)`).

`DatasetRef` is the one way any config names shards — the pretrainer and the
decomposition trainer share it. It lives beside the layout because resolving its `name`
arm IS the layout.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, Self

from pydantic import Discriminator, Field, NonNegativeInt, PositiveInt, model_validator

from param_decomp_formats.config import BaseConfig

DATASET_META_FILENAME = "meta.json"
ROW_SHUFFLE_FILENAME = "row_shuffle.json"


def dataset_dir(data_root: Path, name: str) -> Path:
    """The store layout: a named dataset's shards live at `<data_root>/datasets/<name>`."""
    return data_root / "datasets" / name


class NamedDataset(BaseConfig):
    """A dataset in the store: shards + `meta.json` at `<data_root>/datasets/<name>`.
    Names are immutable versions — a changed dataset is a new name."""

    kind: Literal["name"] = "name"
    name: str

    @model_validator(mode="after")
    def _flat_name(self) -> Self:
        assert self.name not in ("", ".", "..") and not any(
            char in self.name for char in "/\\*?["
        ), f"dataset names are flat store names: {self.name!r}"
        return self


class DatasetDir(BaseConfig):
    """Ad-hoc escape hatch: an explicit directory of `*.parquet` shards. Machine-specific
    by nature, so the path must be absolute; a named store dataset is the portable form."""

    kind: Literal["dir"] = "dir"
    dir: Path

    @model_validator(mode="after")
    def _absolute(self) -> Self:
        assert self.dir.is_absolute(), f"ad-hoc shard dirs are absolute paths: {self.dir}"
        return self


DatasetRef = Annotated[NamedDataset | DatasetDir, Discriminator("kind")]

DataLayout = Literal["documents", "unsegmented_tokens"]
"""How a run reads its corpus's rows. `documents`: attention and next-token terms stay
within each document. `unsegmented_tokens`: each row is one unbroken sequence, its document
IDs set aside. Either way a new run's corpus records document IDs
(`assert_records_documents`). A target is read the way it was trained: the layout belongs
to the run, not the shards."""


def resolve_dataset_ref(ref: DatasetRef, data_root: Path) -> Path:
    match ref:
        case NamedDataset(name=name):
            return dataset_dir(data_root, name)
        case DatasetDir(dir=dir):
            return dir


class DatasetIdentity(BaseConfig):
    """Tokenizer and staged row width, independent of the shard storage format.

    Enough to tokenize new prompts or decode stored token IDs; this does not certify
    that a dataset can be read by the current training loader.
    """

    seq_len: PositiveInt
    tokenizer_name: str = Field(min_length=1)


class DatasetMeta(DatasetIdentity):
    """`seq_len` is the exact staged row width, including any final next-token label.
    `tokenizer_name` is the tokenizer that produced the ids — the decode authority for
    consumers rendering harvested tokens. Version 2 requires paired int32 `input_ids`
    and `document_ids` columns. `preprocessing_name` records the producing recipe;
    it is provenance, not a runtime behavior switch. Older artifacts must be tokenized again.
    `end_of_document_token_id` is the token that ends every document, or None for datasets
    staged before it was recorded; it is not derivable from the tokenizer (Qwen3 documents end
    in `<|endoftext|>`, not its EOS `<|im_end|>`)."""

    format_version: Literal[2]
    preprocessing_name: str = Field(min_length=1)
    end_of_document_token_id: NonNegativeInt | None = None


def read_dataset_identity(data_dir: Path) -> DatasetIdentity:
    """Read only tokenizer/context facts; never authorize reading or rewriting shards."""
    raw = json.loads((data_dir / DATASET_META_FILENAME).read_text())
    return DatasetIdentity(seq_len=raw["seq_len"], tokenizer_name=raw["tokenizer_name"])


def read_dataset_meta(data_dir: Path) -> DatasetMeta:
    path = data_dir / DATASET_META_FILENAME
    assert path.exists(), (
        f"no {DATASET_META_FILENAME} in {data_dir}: dataset dirs are self-describing "
        "(prestage writes document-aware metadata; older datasets must be tokenized again)"
    )
    return DatasetMeta.model_validate_json(path.read_text())


def assert_records_documents(data_dir: Path) -> DatasetMeta:
    """Refuse a dataset without document IDs: training and pretraining read only
    document-aware datasets. Older token-only datasets are read only to reopen finished runs
    that trained on them (`migrations.data_layout`)."""
    path = data_dir / DATASET_META_FILENAME
    assert not path.exists() or "format_version" in json.loads(path.read_text()), (
        f"{data_dir} is a token-only dataset with no document IDs; new runs read only "
        "document-aware datasets"
    )
    return read_dataset_meta(data_dir)


def assert_row_shuffled(data_dir: Path) -> None:
    """Refuse a dataset whose rows are not recorded as uniformly shuffled across it.

    Packed corpora store each document's rows next to each other, and documents run to many
    rows, so a contiguous range of rows is a handful of documents. The training loader cuts a
    batch from one or two row groups, and the eval loader reads the dataset's first rows: both
    read a sample of the corpus only if its rows were shuffled across it.
    `row_shuffle.json` records that they were and how (`infra.row_shuffle`)."""
    assert (data_dir / ROW_SHUFFLE_FILENAME).exists(), (
        f"{data_dir} records no row shuffle ({ROW_SHUFFLE_FILENAME}): its rows may be in "
        "packing order, where a contiguous range is a few long documents, not a sample. Read "
        "its row-shuffled copy, or tokenize it again with a producer that ends in "
        "infra.row_shuffle, e.g. prestage_tokenized tokenize then finish."
    )


def write_dataset_meta(data_dir: Path, meta: DatasetMeta) -> None:
    """Publish immutable metadata atomically, including concurrent prestage workers."""
    path = data_dir / DATASET_META_FILENAME
    with NamedTemporaryFile(mode="w", dir=data_dir.parent, suffix=".meta.json") as temporary:
        temporary.write(meta.model_dump_json() + "\n")
        temporary.flush()
        try:
            path.hardlink_to(temporary.name)
        except FileExistsError:
            assert read_dataset_meta(data_dir) == meta, f"dataset metadata differs in {data_dir}"
