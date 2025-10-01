import re
from collections import defaultdict
from typing import Any, cast, override

import torch
import wandb
from jaxtyping import Float
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_ci_l_zero


class CI_L0(Metric):
    """L0 metric for CI values.

    NOTE: Assumes all batches and sequences are the same size.
    """

    is_differentiable: bool | None = False

    l0_values: dict[str, list[Tensor] | Tensor]

    def __init__(
        self,
        model: ComponentModel,
        ci_alive_threshold: float,
        groups: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.l0_threshold = ci_alive_threshold
        self.groups = groups

        all_keys = model.module_paths
        if groups:
            all_keys += list(groups.keys())

        self.add_state("l0_values", default={key: [] for key in all_keys}, dist_reduce_fx="cat")

    @override
    def update(self, *, ci: dict[str, Float[Tensor, "... C"]], **_: Any) -> None:
        group_sums = defaultdict(float) if self.groups else {}
        for layer_name, layer_ci in ci.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            cast(list[Tensor], self.l0_values[layer_name]).append(torch.tensor([l0_val]))

            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break
        for group_name, group_sum in group_sums.items():
            cast(list[Tensor], self.l0_values[group_name]).append(torch.tensor([group_sum]))

    @override
    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        out = {}
        table_data = []
        for key, l0s in self.l0_values.items():
            # After sync_dist(), l0s is a single tensor; otherwise it's a list of tensors
            l0_tensor = torch.cat(l0s, dim=0) if isinstance(l0s, list) else l0s
            avg_l0 = l0_tensor.mean().item()
            out[f"l0_{self.l0_threshold}/{key}"] = avg_l0
            table_data.append((key, avg_l0))
        bar_chart = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{self.l0_threshold}",
        )
        out["l0_bar_chart"] = bar_chart
        return out
