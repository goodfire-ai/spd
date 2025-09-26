from typing import Any, override

import torch
from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.models.component_model import ComponentModel
from spd.plotting import plot_mean_component_cis_both_scales


class CIMeanPerComponent(Metric):
    slow = True
    is_differentiable: bool | None = False
    full_state_update: bool | None = False  # Avoid double update calls

    def __init__(self, model: ComponentModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.components = model.components

        for module_name in self.components:
            self.add_state(
                f"component_ci_sums_{module_name}",
                default=torch.zeros(model.C),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"examples_seen_{module_name}",
                default=torch.tensor(0),
                dist_reduce_fx="sum",
            )

    @override
    def update(self, *, ci: dict[str, Tensor], **_: Any) -> None:
        for module_name, ci_vals in ci.items():
            n_leading_dims = ci_vals.ndim - 1
            n_examples = ci_vals.shape[:n_leading_dims].numel()

            examples_seen = getattr(self, f"examples_seen_{module_name}")
            examples_seen += n_examples

            ci_sums = getattr(self, f"component_ci_sums_{module_name}")
            leading_dim_idxs = tuple(range(n_leading_dims))
            ci_sums += ci_vals.sum(dim=leading_dim_idxs)

    @override
    def compute(self) -> dict[str, Image.Image]:
        mean_component_cis = {}
        for module_name in self.components:
            ci_sums = getattr(self, f"component_ci_sums_{module_name}")
            examples_seen = getattr(self, f"examples_seen_{module_name}")
            mean_component_cis[module_name] = ci_sums / examples_seen

        img_linear, img_log = plot_mean_component_cis_both_scales(mean_component_cis)

        return {
            "figures/ci_mean_per_component": img_linear,
            "figures/ci_mean_per_component_log": img_log,
        }
