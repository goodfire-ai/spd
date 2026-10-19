"""Read the stable training facts needed by offline consumers of an LM run.

`load_deliverable` resolves target structure, CI definition, datasets, and seed from the
finished run without restoring its checkpoint. It accepts the normalized deliverable file
and the current launch config used by older runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from param_decomp_formats.config import BaseConfig
from param_decomp_formats.dataset_store import resolve_dataset_ref
from pydantic import ConfigDict, TypeAdapter

from param_decomp.experiments.lm.config import (
    DenseCSpec,
    LMDecompositionConfig,
    LMTargetConfig,
    resolve_decomposition,
    resolve_dense_decomposition,
    resolve_lm_ci_fn_arch,
)
from param_decomp.experiments.lm.resolved import AnyLMTargetConfig, LMCIFnArch, ResolvedLMData
from param_decomp.experiments.lm.run_data import LMDataConfig, product_document
from param_decomp.migrations.data_layout import migrate_raw as migrate_data_layout


class _ScheduleSeed(BaseConfig):
    model_config = ConfigDict(extra="ignore")

    seed: int


@dataclass(frozen=True)
class ResolvedDeliverable:
    """Target, CI definition, datasets, and seed fixed by a finished training run."""

    target: AnyLMTargetConfig
    ci_fn: LMCIFnArch
    data: ResolvedLMData
    seed: int


def _mapping(raw: object, field_for_err: str) -> dict[str, Any]:
    assert isinstance(raw, dict), f"stored run {field_for_err} must be a mapping"
    return raw


def load_deliverable(run_dir: Path, data_root: Path) -> ResolvedDeliverable:
    """Resolve the current product schema from a normalized product or current run pin."""
    raw = _mapping(yaml.safe_load(product_document(run_dir).read_text()), "config")
    raw = migrate_data_layout(raw, data_root)
    target_raw = _mapping(raw.get("target"), "target")
    target_config = LMTargetConfig.model_validate(target_raw)
    decomposition = LMDecompositionConfig.model_validate(
        _mapping(raw.get("decomposition"), "decomposition")
    )
    data = LMDataConfig.model_validate(_mapping(raw.get("data"), "data"))
    schedule = _ScheduleSeed.model_validate(_mapping(raw.get("pd"), "pd"))

    resolved = resolve_decomposition(target_config, decomposition, data_root)
    ci_fn = resolve_lm_ci_fn_arch(resolved, decomposition.ci)
    return ResolvedDeliverable(
        target=resolved.target,
        ci_fn=ci_fn,
        data=_resolved_data(data, data_root),
        seed=schedule.seed,
    )


def _resolved_data(data: LMDataConfig, data_root: Path) -> ResolvedLMData:
    return ResolvedLMData(
        dir=resolve_dataset_ref(data.train, data_root),
        eval_dir=resolve_dataset_ref(data.eval, data_root),
        layout=data.layout,
        seq_len=data.seq_len,
    )


def load_dense_target(run_dir: Path, data_root: Path) -> AnyLMTargetConfig:
    """The target of a dense-transformer run, resolved from exactly two sections of its
    stored config: `target` and `decomposition.sites`, each validated strictly by its own
    type. The CI definition is never read; consumers that substitute component activations
    have no use for it."""
    raw = _mapping(yaml.safe_load(product_document(run_dir).read_text()), "config")
    target_config = LMTargetConfig.model_validate(_mapping(raw["target"], "target"))
    decomposition = _mapping(raw["decomposition"], "decomposition")
    sites = TypeAdapter(DenseCSpec).validate_python(decomposition["sites"])
    return resolve_dense_decomposition(target_config, sites, data_root).target
