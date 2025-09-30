from typing import Any, override

import torch
from einops import reduce
from jaxtyping import Int
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import plot_component_activation_density


class ComponentActivationDensity(Metric):
    """Activation density for each component."""

    slow = True
    is_differentiable: bool | None = False

    n_examples: Int[Tensor, ""]
    component_activation_counts: dict[str, Tensor]

    def __init__(self, model: ComponentModel, ci_alive_threshold: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.ci_alive_threshold = ci_alive_threshold

        self.add_state("n_examples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "component_activation_counts",
            default={module_name: torch.zeros(model.C) for module_name in model.components},
            dist_reduce_fx="sum",
        )

    @override
    def update(self, *, ci: dict[str, Tensor], **_: Any) -> None:
        n_examples_this_batch = next(iter(ci.values())).shape[:-1].numel()
        self.n_examples += n_examples_this_batch

        for module_name, ci_vals in ci.items():
            active_components = ci_vals > self.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += (
                n_activations_per_component * n_examples_this_batch
            )

    @override
    def compute(self) -> dict[str, Image.Image]:
        activation_densities = {}
        for module_name in self.model.components:
            activation_densities[module_name] = (
                self.component_activation_counts[module_name] / self.n_examples
            )

        fig = plot_component_activation_density(activation_densities)
        return {"figures/component_activation_density": fig}
