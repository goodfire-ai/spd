from typing import Any, override

import torch
from einops import reduce
from PIL import Image
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import MetricInterface
from spd.models.component_model import ComponentModel
from spd.plotting import plot_component_activation_density
from spd.utils.distributed_utils import all_reduce


class ComponentActivationDensity(MetricInterface):
    """Activation density for each component."""

    def __init__(self, model: ComponentModel, ci_alive_threshold: float, device: str) -> None:
        self.model = model
        self.ci_alive_threshold = ci_alive_threshold
        device = device
        self.n_examples = torch.tensor(0.0, device=device)
        self.component_activation_counts: dict[str, Tensor] = {
            module_name: torch.zeros(model.C, device=device) for module_name in model.components
        }

    @override
    def update(
        self,
        *,
        ci: dict[str, Tensor],
        **_: Any,
    ) -> None:
        n_examples_this_batch = next(iter(ci.values())).shape[:-1].numel()
        self.n_examples += n_examples_this_batch

        for module_name, ci_vals in ci.items():
            active_components = ci_vals > self.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += n_activations_per_component

    @override
    def compute(self) -> dict[str, Image.Image]:
        # Reduce across ranks
        n_examples_reduced = all_reduce(self.n_examples, op=ReduceOp.SUM)
        component_activation_counts_reduced = {
            name: all_reduce(val, op=ReduceOp.SUM)
            for name, val in self.component_activation_counts.items()
        }

        activation_densities = {}
        for module_name in self.model.components:
            activation_densities[module_name] = (
                component_activation_counts_reduced[module_name] / n_examples_reduced
            )

        fig = plot_component_activation_density(activation_densities)
        return {"figures/component_activation_density": fig}
