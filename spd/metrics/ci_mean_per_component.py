from typing import Any, override

import torch
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import plot_mean_component_cis_both_scales


class CIMeanPerComponent(Metric):
    is_differentiable: bool | None = False

    component_ci_sums: dict[str, Tensor]
    examples_seen: dict[str, Tensor]

    def __init__(self, model: ComponentModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.components = model.components

        self.add_state(
            "component_ci_sums",
            default={module_name: torch.zeros(model.C) for module_name in self.components},
            dist_reduce_fx="sum",
        )
        self.add_state(
            "examples_seen",
            default={module_name: torch.tensor(0) for module_name in self.components},
            dist_reduce_fx="sum",
        )

    @override
    def update(self, *, ci: dict[str, Tensor], **_: Any) -> None:
        for module_name, ci_vals in ci.items():
            n_leading_dims = ci_vals.ndim - 1
            n_examples = ci_vals.shape[:n_leading_dims].numel()

            self.examples_seen[module_name] += n_examples

            leading_dim_idxs = tuple(range(n_leading_dims))
            self.component_ci_sums[module_name] += ci_vals.sum(dim=leading_dim_idxs)

    @override
    def compute(self) -> dict[str, Image.Image]:
        mean_component_cis = {}
        for module_name in self.components:
            mean_component_cis[module_name] = (
                self.component_ci_sums[module_name] / self.examples_seen[module_name]
            )

        img_linear, img_log = plot_mean_component_cis_both_scales(mean_component_cis)

        return {
            "figures/ci_mean_per_component": img_linear,
            "figures/ci_mean_per_component_log": img_log,
        }
