"""Data types for harvest pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.settings import SPD_OUT_DIR

# Base directory for harvest data
HARVEST_DATA_DIR = SPD_OUT_DIR / "harvest"


def get_harvest_dir(wandb_run_id: str) -> Path:
    """Get the base harvest directory for a run."""
    return HARVEST_DATA_DIR / wandb_run_id


def get_activation_contexts_dir(wandb_run_id: str) -> Path:
    """Get the activation contexts directory for a run."""
    return get_harvest_dir(wandb_run_id) / "activation_contexts"


def get_correlations_dir(wandb_run_id: str) -> Path:
    """Get the correlations directory for a run."""
    return get_harvest_dir(wandb_run_id) / "correlations"


@dataclass
class ActivationExample:
    token_ids: list[int]
    ci_values: list[float]
    component_acts: list[float]  # Normalized component activations: (v_i^T @ a) * ||u_i||


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
