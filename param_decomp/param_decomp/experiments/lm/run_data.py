"""A finished LM run's data config, read from its run directory without loading JAX."""

from pathlib import Path

import yaml
from param_decomp_formats.config import BaseConfig
from param_decomp_formats.dataset_store import DataLayout, DatasetRef
from pydantic import PositiveInt, TypeAdapter

from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME

DELIVERABLE_FILENAME = "deliverable.yaml"


class LMDataConfig(BaseConfig):
    """The run's data: `train` feeds the trainer; `eval` is the held-out split the eval
    pass reads; `layout` says how both are read, `seq_len` tokens per row. A stored row may
    hold one more token, a pretraining label, which is dropped. Each ref's own facts
    (stored width, tokenizer) ride with its shards as `meta.json`."""

    train: DatasetRef
    eval: DatasetRef
    layout: DataLayout
    seq_len: PositiveInt


def product_document(run_dir: Path) -> Path:
    normalized = run_dir / DELIVERABLE_FILENAME
    return normalized if normalized.is_file() else run_dir / LAUNCH_CONFIG_FILENAME


def load_training_dataset(run_dir: Path) -> DatasetRef:
    """The dataset a finished run trained on, as its product names it."""
    raw = yaml.safe_load(product_document(run_dir).read_text())
    assert isinstance(raw, dict), f"stored run config must be a mapping: {run_dir}"
    return TypeAdapter(DatasetRef).validate_python(raw["data"]["train"])
