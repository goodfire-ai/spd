"""Loaders for reading harvest output files."""

import json

from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
    get_activation_contexts_dir,
    get_correlations_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage


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
                ActivationExample(
                    token_ids=ex["token_ids"],
                    ci_values=ex["ci_values"],
                    inner_acts=ex.get("inner_acts", [0.0] * len(ex["token_ids"])),
                )
                for ex in data["activation_examples"]
            ]
            data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
            data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
            comp = ComponentData(**data)
            components[comp.component_key] = comp
    return components


def load_correlations(wandb_run_id: str) -> CorrelationStorage | None:
    """Load component correlations from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "component_correlations.pt"
    if not path.exists():
        return None
    return CorrelationStorage.load(path)


def load_token_stats(wandb_run_id: str) -> TokenStatsStorage | None:
    """Load token statistics from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "token_stats.pt"
    if not path.exists():
        return None
    return TokenStatsStorage.load(path)
