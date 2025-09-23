from collections import defaultdict
from collections.abc import Mapping
from typing import Any, override

import torch
from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.plotting import plot_ci_values_histograms


class CIHistograms(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(
        self, model: Any, config: Config, n_batches_accum: int | None = None, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.n_batches_accum = n_batches_accum
        self.causal_importances: dict[str, list[Tensor]] = defaultdict(list)
        self.batches_seen: int = 0
        self.model = model

        for module_name in model.components:
            self.add_state(
                f"causal_importances_{module_name}",
                default=[],
                dist_reduce_fx="cat",
            )

    @override
    def update(self, ci: dict[str, Tensor], **kwargs: Any) -> None:
        if self.n_batches_accum is not None and self.batches_seen >= self.n_batches_accum:
            return
        self.batches_seen += 1
        for k, v in ci.items():
            self.causal_importances[k].append(v.detach().cpu())

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        combined_causal_importances = (
            {k: torch.cat(v) for k, v in self.causal_importances.items()}
            if self.causal_importances
            else {}
        )
        fig = plot_ci_values_histograms(causal_importances=combined_causal_importances)
        return {"figures/causal_importance_values": fig}
