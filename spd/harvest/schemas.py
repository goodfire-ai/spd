"""Data types for harvest pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.settings import SPD_OUT_DIR

# Base directory for harvest data
HARVEST_DATA_DIR = SPD_OUT_DIR / "harvest"


def get_harvest_dir(wandb_run_id: str) -> Path:
    """Get the base harvest directory for a run."""
    return HARVEST_DATA_DIR / wandb_run_id


def get_harvest_subrun_dir(wandb_run_id: str, subrun_id: str) -> Path:
    """Get the sub-run directory for a specific harvest invocation."""
    return get_harvest_dir(wandb_run_id) / subrun_id


@dataclass
class ActivationExample:
    token_ids: list[int]
    ci_values: list[float]
    component_acts: list[float]  # Normalized component activations: (v_i^T @ a) * ||u_i||

    def __post_init__(self) -> None:
        self._strip_legacy_padding()

    def _strip_legacy_padding(self) -> None:
        """Strip -1 padding sentinels from old harvest data.

        Old harvests padded token windows with -1 at sequence boundaries.
        New harvests strip at write time (harvester.py), but existing data on disk
        still has them. Remove once all harvest data is regenerated.
        """
        PAD = -1
        if any(t == PAD for t in self.token_ids):
            mask = [t != PAD for t in self.token_ids]
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


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int
    mean_ci: float
    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI
