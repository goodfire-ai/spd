"""Minimal activation extraction for component dashboard.

Stripped down version with only what's needed for component data generation.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel, OutputWithCache


class SubcomponentLabel(NamedTuple):
    """Label for a single component: (module_name, component_index)."""

    module: str
    index: int

    def to_string(self) -> str:
        """Convert to string format 'module:index'."""
        return f"{self.module}:{self.index}"

    @classmethod
    def from_string(cls, label: str) -> "SubcomponentLabel":
        """Parse from string format 'module:index'."""
        module: str
        index_str: str
        module, index_str = label.rsplit(":", 1)
        return cls(module=module, index=int(index_str))


def component_activations(
    model: ComponentModel,
    device: torch.device | str,
    batch: Int[Tensor, "batch_size n_ctx"],
) -> dict[str, Float[Tensor, "batch*n_ctx C"]]:
    """Get component activations over a single batch.

    Args:
        model: ComponentModel to extract activations from
        device: Device for computation
        batch: Input token IDs

    Returns:
        Dict mapping module names to activation tensors [batch*n_ctx, n_components]
    """
    with torch.no_grad():
        model_output: OutputWithCache = model(
            batch.to(device),
            cache_type="input",
        )

        causal_importances: dict[str, Float[Tensor, "steps C"]] = model.calc_causal_importances(
            pre_weight_acts=model_output.cache,
            sampling="continuous",
            detach_inputs=False,
        ).upper_leaky

    return causal_importances


@dataclass(frozen=True)
class ProcessedActivations:
    """Concatenated activations with component labels.

    Simple structure - just concatenates all module activations and tracks labels.
    """

    activations: Float[Tensor, "samples n_components"]
    labels: list[SubcomponentLabel]  # Component labels as (module, index) tuples

    def __post_init__(self) -> None:
        """Validate shape consistency."""
        assert self.activations.shape[1] == len(self.labels), (
            f"Activations shape {self.activations.shape[1]} != labels length {len(self.labels)}"
        )

    @property
    def n_components(self) -> int:
        """Total number of components."""
        return len(self.labels)

    @property
    def label_strings(self) -> list[str]:
        """Get labels as strings in 'module:index' format."""
        return [label.to_string() for label in self.labels]


def process_activations(
    activations: dict[str, Float[Tensor, "batch*n_ctx C"]],
) -> ProcessedActivations:
    """Concatenate all module activations into single matrix with labels.

    Args:
        activations: Dict mapping module names to activation tensors

    Returns:
        ProcessedActivations with concatenated activations and labels
    """
    # Build labels and concatenate activations
    labels: list[SubcomponentLabel] = []
    acts_list: list[Float[Tensor, "samples C"]] = []

    for module_name in sorted(activations.keys()):
        acts: Float[Tensor, "samples C"] = activations[module_name]
        n_components: int = acts.shape[1]

        # Add labels as SubcomponentLabel tuples
        labels.extend([SubcomponentLabel(module=module_name, index=i) for i in range(n_components)])
        acts_list.append(acts)

    # Concatenate all activations
    concatenated: Float[Tensor, "samples n_components"] = torch.cat(acts_list, dim=1)

    return ProcessedActivations(
        activations=concatenated,
        labels=labels,
    )
