from typing import Any, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from spd.metrics.base import MetricInterface
from spd.models.component_model import ComponentModel
from spd.plotting import plot_ci_values_histograms
from spd.utils.distributed_utils import gather_all_tensors


class CIHistograms(MetricInterface):
    def __init__(
        self,
        model: ComponentModel,
        n_batches_accum: int | None = None,
    ):
        self.n_batches_accum = n_batches_accum
        self.module_names = list(model.components.keys())
        self.batches_seen = 0
        self.causal_importances: dict[str, list[Tensor]] = {
            module_name: [] for module_name in self.module_names
        }

    @override
    def update(
        self,
        *,
        ci: dict[str, Tensor],
        **_: Any,
    ) -> None:
        if self.n_batches_accum is not None and self.batches_seen >= self.n_batches_accum:
            return
        self.batches_seen += 1
        for k, v in ci.items():
            self.causal_importances[k].append(v)

    @override
    def compute(self) -> dict[str, Image.Image]:
        cis: dict[str, Float[Tensor, "... C"]] = {}
        for module_name in self.module_names:
            ci_list = self.causal_importances[module_name]
            cis[module_name] = torch.cat(gather_all_tensors(torch.cat(ci_list, dim=0)), dim=0)

        fig = plot_ci_values_histograms(causal_importances=cis)
        return {"figures/causal_importance_values": fig}
