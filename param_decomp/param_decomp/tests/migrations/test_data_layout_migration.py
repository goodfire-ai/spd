"""Runs pinned before `data.layout` and `data.seq_len` load with their trainer's choices."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from param_decomp_formats.dataset_store import DatasetMeta, dataset_dir, write_dataset_meta

from param_decomp.experiments.lm.config import load_config
from param_decomp.migrations.data_layout import migrate_raw
from param_decomp.tests.lm_configs import l18_mlp_raw


def _stage(data_root: Path, name: str, seq_len: int) -> None:
    directory = dataset_dir(data_root, name)
    directory.mkdir(parents=True)
    write_dataset_meta(
        directory,
        DatasetMeta(seq_len=seq_len, tokenizer_name="t", format_version=2, preprocessing_name="p"),
    )


def _pinned_before_layouts(data_root: Path) -> dict[str, Any]:
    raw = l18_mlp_raw()
    raw["data"] = {
        "train": {"kind": "name", "name": "train-v1"},
        "eval": {"kind": "name", "name": "eval-v1"},
    }
    _stage(data_root, "train-v1", 513)
    _stage(data_root, "eval-v1", 513)
    return raw


@pytest.mark.parametrize(
    ("sites", "layout"),
    [
        ("glu_transformer", "documents"),
        ("simple_mlp", "unsegmented_tokens"),
        ("qwen36_moe", "unsegmented_tokens"),
    ],
)
def test_layout_follows_the_old_layer_type_rule_and_width_the_metadata(
    tmp_path: Path, sites: str, layout: str
) -> None:
    raw = _pinned_before_layouts(tmp_path)
    raw["decomposition"]["sites"]["kind"] = sites
    migrated = migrate_raw(raw, tmp_path)
    assert migrated["data"]["layout"] == layout and migrated["data"]["seq_len"] == 513
    assert "layout" not in raw["data"]


def test_declared_data_is_left_alone(tmp_path: Path) -> None:
    raw = l18_mlp_raw()
    assert migrate_raw(raw, tmp_path) is raw


def test_a_run_pinned_before_layouts_loads_as_a_fine_tune_parent(tmp_path: Path) -> None:
    """`load_config` is the fine-tune parent check's reader; a pre-layout pin must build."""
    raw = _pinned_before_layouts(tmp_path)
    pinned = tmp_path / "launch_config.yaml"
    pinned.write_text(yaml.safe_dump(raw))

    built, authored = load_config(pinned, "p-0123abcd", tmp_path)

    assert authored.data.layout == "documents" and built.data.seq_len == 513
