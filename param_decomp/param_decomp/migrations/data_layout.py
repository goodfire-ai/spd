"""Give a stored LM run's `data` the layout and width its trainer used.

Runs launched before `data.layout` and `data.seq_len` existed chose both at startup: the
layout from the target's layer types (plain-MLP and Qwen3.6-MoE targets read whole rows,
GLU transformers read documents) and the width from the training dataset's `meta.json`.
``migrate_raw`` writes those choices into a complete stored experiment config in memory.
The LM loaders apply it at their stored-config boundary, so immutable pinned configs still
load, resume and serve as fine-tune parents; newly authored configs must declare both.
"""

import copy
from pathlib import Path
from typing import Any

from param_decomp_formats.dataset_store import (
    DataLayout,
    DatasetRef,
    read_dataset_identity,
    resolve_dataset_ref,
)
from pydantic import TypeAdapter

_LAYOUT_BY_SITES: dict[str, DataLayout] = {
    "simple_mlp": "unsegmented_tokens",
    "qwen36_moe": "unsegmented_tokens",
    "glu_transformer": "documents",
}


def migrate_raw(raw: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """`raw` with `data.layout` and `data.seq_len` filled in where they are absent."""
    data = raw.get("data")
    if not isinstance(data, dict) or "layout" in data:
        return raw
    migrated = copy.deepcopy(raw)
    train = TypeAdapter(DatasetRef).validate_python(migrated["data"]["train"])
    migrated["data"]["layout"] = _LAYOUT_BY_SITES[migrated["decomposition"]["sites"]["kind"]]
    migrated["data"]["seq_len"] = read_dataset_identity(
        resolve_dataset_ref(train, data_root)
    ).seq_len
    return migrated
