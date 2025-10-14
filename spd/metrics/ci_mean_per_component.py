from typing import Any, ClassVar, override

import torch
from PIL import Image
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.plotting import plot_mean_component_cis_both_scales
from spd.utils.distributed_utils import all_reduce


class CIMeanPerComponent(Metric):
    slow: ClassVar[bool] = True

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.components = model.components
        self.component_ci_sums: dict[str, Tensor] = {
            module_name: torch.zeros(model.C, device=device) for module_name in self.components
        }
        self.examples_seen: dict[str, Tensor] = {
            module_name: torch.tensor(0, device=device) for module_name in self.components
        }

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        for module_name, ci_vals in ci.lower_leaky.items():
            n_leading_dims = ci_vals.ndim - 1
            n_examples = ci_vals.shape[:n_leading_dims].numel()

            self.examples_seen[module_name] += n_examples

            leading_dim_idxs = tuple(range(n_leading_dims))
            self.component_ci_sums[module_name] += ci_vals.sum(dim=leading_dim_idxs)

    @override
    def compute(self) -> dict[str, Image.Image]:
        mean_component_cis = {}
        for module_name in self.components:
            summed_ci = all_reduce(self.component_ci_sums[module_name], op=ReduceOp.SUM)
            examples_reduced = all_reduce(self.examples_seen[module_name], op=ReduceOp.SUM)
            mean_component_cis[module_name] = summed_ci / examples_reduced

        img_linear, img_log = plot_mean_component_cis_both_scales(mean_component_cis)

        return {
            "figures/ci_mean_per_component": img_linear,
            "figures/ci_mean_per_component_log": img_log,
        }
