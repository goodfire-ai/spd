"""Data types for harvest pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.settings import REPO_ROOT

# Base directory for all harvest data
HARVEST_DATA_DIR = REPO_ROOT / ".data" / "harvest"


def get_harvest_dir(wandb_run_id: str) -> Path:
    """Get the base harvest directory for a run."""
    return HARVEST_DATA_DIR / wandb_run_id


def get_activation_contexts_dir(wandb_run_id: str) -> Path:
    """Get the activation contexts directory for a run."""
    return get_harvest_dir(wandb_run_id) / "activation_contexts"


def get_correlations_dir(wandb_run_id: str) -> Path:
    """Get the correlations directory for a run."""
    return get_harvest_dir(wandb_run_id) / "correlations"


def get_autointerp_dir(wandb_run_id: str) -> Path:
    """Get the autointerp (interpretations) directory for a run."""
    return get_harvest_dir(wandb_run_id) / "autointerp"


@dataclass
class ActivationExample:
    token_ids: list[int]
    ci_values: list[float]


@dataclass
class ComponentTokenPMI:
    top: list[tuple[int, float]]
    bottom: list[tuple[int, float]]


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int
    mean_ci: float
    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI
