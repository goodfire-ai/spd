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
    activation_values: list[float]
    component_acts: list[float]  # Normalized component activations: (v_i^T @ a) * ||u_i||

    def __init__(
        self,
        token_ids: list[int],
        activation_values: list[float] | None = None,
        component_acts: list[float] | None = None,
        ci_values: list[float] | None = None,
    ) -> None:
        # Keep backward compatibility with old CI-named payloads.
        values = activation_values if activation_values is not None else ci_values
        assert values is not None, "activation_values (or legacy ci_values) is required"
        assert component_acts is not None, "component_acts is required"
        self.token_ids = token_ids
        self.activation_values = values
        self.component_acts = component_acts
        self.__post_init__()

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
            self.activation_values = [
                v for v, k in zip(self.activation_values, mask, strict=True) if k
            ]
            self.component_acts = [v for v, k in zip(self.component_acts, mask, strict=True) if k]

    @property
    def ci_values(self) -> list[float]:
        """Backward-compatible alias for activation_values."""
        return self.activation_values

    @ci_values.setter
    def ci_values(self, value: list[float]) -> None:
        self.activation_values = value


@dataclass
class ComponentTokenPMI:
    top: list[tuple[int, float]]
    bottom: list[tuple[int, float]]


@dataclass
class ComponentSummary:
    """Lightweight summary of a component (for /summary endpoint)."""

    layer: str
    component_idx: int
    mean_activation: float

    def __init__(
        self,
        layer: str,
        component_idx: int,
        mean_activation: float | None = None,
        mean_ci: float | None = None,
    ) -> None:
        mean = mean_activation if mean_activation is not None else mean_ci
        assert mean is not None, "mean_activation (or legacy mean_ci) is required"
        self.layer = layer
        self.component_idx = component_idx
        self.mean_activation = mean

    @property
    def mean_ci(self) -> float:
        """Backward-compatible alias for mean_activation."""
        return self.mean_activation

    @mean_ci.setter
    def mean_ci(self, value: float) -> None:
        self.mean_activation = value


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int
    mean_activation: float
    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI

    def __init__(
        self,
        component_key: str,
        layer: str,
        component_idx: int,
        activation_examples: list[ActivationExample],
        input_token_pmi: ComponentTokenPMI,
        output_token_pmi: ComponentTokenPMI,
        mean_activation: float | None = None,
        mean_ci: float | None = None,
    ) -> None:
        mean = mean_activation if mean_activation is not None else mean_ci
        assert mean is not None, "mean_activation (or legacy mean_ci) is required"
        self.component_key = component_key
        self.layer = layer
        self.component_idx = component_idx
        self.mean_activation = mean
        self.activation_examples = activation_examples
        self.input_token_pmi = input_token_pmi
        self.output_token_pmi = output_token_pmi

    @property
    def mean_ci(self) -> float:
        """Backward-compatible alias for mean_activation."""
        return self.mean_activation

    @mean_ci.setter
    def mean_ci(self, value: float) -> None:
        self.mean_activation = value
