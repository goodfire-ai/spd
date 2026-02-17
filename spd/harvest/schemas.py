"""Data types for harvest pipeline."""

from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.settings import SPD_OUT_DIR

# Base directory for harvest data
HARVEST_DATA_DIR = SPD_OUT_DIR / "harvest"


def get_harvest_dir(wandb_run_id: str) -> Path:
    """Get the base harvest directory for a run."""
    return HARVEST_DATA_DIR / wandb_run_id


def get_harvest_subrun_dir(decomposition_id: str, subrun_id: str) -> Path:
    """Get the sub-run directory for a specific harvest invocation."""
    return get_harvest_dir(decomposition_id) / subrun_id


@dataclass
class HarvestBatch:
    """Output of a method-specific harvest function for a single batch.

    The harvest loop calls the user-provided harvest_fn on each raw dataloader batch,
    which returns one of these. The harvest loop then feeds it to the Harvester.

    firings/activations are keyed by layer name. activations values are keyed by
    activation type (e.g. "causal_importance", "component_activation" for SPD;
    just "activation" for SAEs).
    """

    tokens: Int[Tensor, "batch seq"]
    firings: dict[str, Bool[Tensor, "batch seq c"]]
    activations: dict[str, dict[str, Float[Tensor, "batch seq c"]]]
    output_probs: Float[Tensor, "batch seq vocab"]


@dataclass
class ActivationExample:
    """Activation example for a single component. no padding"""

    token_ids: list[int]
    firings: list[bool]
    activations: dict[str, list[float]]

    def __post_init__(self) -> None:
        self._strip_legacy_padding()

    def _strip_legacy_padding(self) -> None:
        """Strip -1 padding sentinels from old harvest data."""
        PAD = -1
        if any(t == PAD for t in self.token_ids):
            mask = [t != PAD for t in self.token_ids]
            self.token_ids = [v for v, k in zip(self.token_ids, mask, strict=True) if k]
            self.firings = [v for v, k in zip(self.firings, mask, strict=True) if k]
            for act_type in self.activations:
                self.activations[act_type] = [
                    v for v, k in zip(self.activations[act_type], mask, strict=True) if k
                ]


@dataclass
class ComponentTokenPMI:
    top: list[tuple[int, float]]
    bottom: list[tuple[int, float]]


@dataclass
class ComponentSummary:
    """Lightweight summary of a component (for /summary endpoint)."""

    layer: str
    component_idx: int
    firing_density: float
    mean_activations: dict[str, float]
    """Key is activation type, (e.g. "causal_importance", "component_activation", etc.)"""


@dataclass
class ComponentData:
    component_key: str
    layer: str
    component_idx: int

    mean_activations: dict[str, float]
    firing_density: float

    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI
