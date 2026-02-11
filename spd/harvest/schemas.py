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


_PAD_SENTINEL = -1


@dataclass
class ActivationExample:
    token_ids: list[int]
    ci_values: list[float]
    component_acts: list[float]  # Normalized component activations: (v_i^T @ a) * ||u_i||

    def __post_init__(self) -> None:
        if any(t == _PAD_SENTINEL for t in self.token_ids):
            mask = [t != _PAD_SENTINEL for t in self.token_ids]
            self.token_ids = [v for v, k in zip(self.token_ids, mask, strict=True) if k]
            self.ci_values = [v for v, k in zip(self.ci_values, mask, strict=True) if k]
            self.component_acts = [v for v, k in zip(self.component_acts, mask, strict=True) if k]


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
