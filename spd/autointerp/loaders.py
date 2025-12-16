"""Loaders for reading harvest output files.

These loaders provide a clean interface for the app to read harvest data.
"""

import json
from dataclasses import dataclass

from spd.autointerp.harvest import ComponentCorrelations, ComponentTokenStats
from spd.autointerp.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
    InterpretationResult,
    get_activation_contexts_dir,
    get_autointerp_dir,
    get_correlations_dir,
)


@dataclass
class HarvestData:
    """Lightweight handle for checking harvest data existence."""

    wandb_run_id: str

    def has_activation_contexts(self) -> bool:
        path = get_activation_contexts_dir(self.wandb_run_id) / "components.jsonl"
        return path.exists()

    def has_correlations(self) -> bool:
        path = get_correlations_dir(self.wandb_run_id) / "component_correlations.pt"
        return path.exists()

    def has_token_stats(self) -> bool:
        path = get_correlations_dir(self.wandb_run_id) / "token_stats.pt"
        return path.exists()

    def has_interpretations(self) -> bool:
        path = get_autointerp_dir(self.wandb_run_id) / "results.jsonl"
        return path.exists()


def load_activation_contexts(wandb_run_id: str) -> dict[str, ComponentData] | None:
    """Load activation contexts from harvest output."""
    ctx_dir = get_activation_contexts_dir(wandb_run_id)
    path = ctx_dir / "components.jsonl"
    if not path.exists():
        return None

    components: dict[str, ComponentData] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            data["activation_examples"] = [
                ActivationExample(**ex) for ex in data["activation_examples"]
            ]
            data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
            data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
            comp = ComponentData(**data)
            components[comp.component_key] = comp
    return components


def load_correlations(wandb_run_id: str) -> ComponentCorrelations | None:
    """Load component correlations from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "component_correlations.pt"
    if not path.exists():
        return None
    return ComponentCorrelations.load(path)


def load_token_stats(wandb_run_id: str) -> ComponentTokenStats | None:
    """Load token statistics from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "token_stats.pt"
    if not path.exists():
        return None
    return ComponentTokenStats.load(path)


def load_interpretations(wandb_run_id: str) -> dict[str, InterpretationResult] | None:
    """Load interpretation results from harvest output."""
    autointerp_dir = get_autointerp_dir(wandb_run_id)
    path = autointerp_dir / "results.jsonl"
    if not path.exists():
        return None

    results: dict[str, InterpretationResult] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            result = InterpretationResult(**data)
            results[result.component_key] = result
    return results
