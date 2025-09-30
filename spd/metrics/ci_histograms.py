from typing import Any, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import plot_ci_values_histograms


class CIHistograms(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        n_batches_accum: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_batches_accum = n_batches_accum
        self.module_names = list(model.components.keys())
        self.batches_seen = 0

        for module_name in self.module_names:
            self.add_state(f"causal_importances_{module_name}", default=[], dist_reduce_fx="cat")

    @override
    def update(self, *, ci: dict[str, Tensor], **_: Any) -> None:
        if self.n_batches_accum is not None and self.batches_seen >= self.n_batches_accum:
            return
        self.batches_seen += 1
        for k, v in ci.items():
            getattr(self, f"causal_importances_{k}").append(v)

    @override
    def compute(self) -> dict[str, Image.Image]:
        cis: dict[str, Float[Tensor, "... C"]] = {}
        for module_name in self.module_names:
            ci_list = getattr(self, f"causal_importances_{module_name}")
            # After self.sync_dist(), this will be a single tensor; otherwise concat list
            cis[module_name] = torch.cat(ci_list, dim=0) if isinstance(ci_list, list) else ci_list

        fig = plot_ci_values_histograms(causal_importances=cis)
        return {"figures/causal_importance_values": fig}
