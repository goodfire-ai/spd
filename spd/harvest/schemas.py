"""Data types for harvest pipeline."""

import json
from dataclasses import asdict, dataclass
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


def load_harvest_ci_threshold(wandb_run_id: str) -> float:
    """Load the CI threshold used during harvest for this run."""
    config_path = get_activation_contexts_dir(wandb_run_id) / "config.json"
    assert config_path.exists(), f"No harvest config at {config_path}"
    with open(config_path) as f:
        return json.load(f)["ci_threshold"]


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
class ComponentSummary:
    """Lightweight summary of a component (for /summary endpoint)."""

    layer: str
    component_idx: int
    mean_ci: float

    @staticmethod
    def save_all(summaries: dict[str, "ComponentSummary"], path: Path) -> None:
        """Save component summaries to JSON file."""
        data = {key: asdict(s) for key, s in summaries.items()}
        path.write_text(json.dumps(data))

    @staticmethod
    def load_all(path: Path) -> dict[str, "ComponentSummary"]:
        """Load component summaries from JSON file."""
        data = json.loads(path.read_text())
        return {key: ComponentSummary(**val) for key, val in data.items()}


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int
    mean_ci: float
    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI
