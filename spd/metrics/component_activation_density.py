from collections.abc import Mapping
from typing import Any, override

import torch
from einops import reduce
from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import plot_component_activation_density


class ComponentActivationDensity(Metric):
    slow = True
    is_differentiable: bool | None = False

    n_examples: int

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config

        self.add_state("n_examples", default=torch.tensor(0.0), dist_reduce_fx="sum")

        for module_name in model.components:
            self.add_state(
                f"component_activation_counts_{module_name}",
                default=torch.zeros(model.C),
                dist_reduce_fx="sum",
            )

    @override
    def update(self, ci: dict[str, Tensor], **kwargs: Any) -> None:
        n_examples_this_batch = next(iter(ci.values())).shape[:-1].numel()
        self.n_examples += n_examples_this_batch

        for module_name, ci_vals in ci.items():
            active_components = ci_vals > self.config.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            counts = getattr(self, f"component_activation_counts_{module_name}")
            counts += n_activations_per_component

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        activation_densities = {}
        for module_name in self.model.components:
            counts = getattr(self, f"component_activation_counts_{module_name}")
            activation_densities[module_name] = counts / self.n_examples

        fig = plot_component_activation_density(activation_densities)
        return {"figures/component_activation_density": fig}
