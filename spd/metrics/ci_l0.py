import re
from collections import defaultdict
from typing import override

import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import MetricInterface
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.distributed_utils import all_reduce, get_device


class CI_L0(MetricInterface):
    """L0 metric for CI values.

    NOTE: Assumes all batches and sequences are the same size.
    """

    def __init__(
        self,
        model: ComponentModel,
        ci_alive_threshold: float,
        groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.l0_threshold = ci_alive_threshold
        self.groups = groups

        all_keys = model.module_paths
        if groups:
            all_keys += list(groups.keys())

        self.l0_values = defaultdict[str, list[float]](list)

    @override
    def update(
        self,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Float[Tensor, "... C"]],
        current_frac_of_training: float,
        ci_upper_leaky: dict[str, Tensor],
        weight_deltas: dict[str, Tensor],
    ) -> None:
        group_sums = defaultdict(float) if self.groups else {}
        for layer_name, layer_ci in ci.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            self.l0_values[layer_name].append(l0_val)

            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break
        for group_name, group_sum in group_sums.items():
            self.l0_values[group_name].append(group_sum)

    @override
    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        out = {}
        table_data = []
        device = get_device()
        for key, l0s in self.l0_values.items():
            # More efficient: sum+count reduction instead of gathering all data
            global_sum = all_reduce(torch.tensor(l0s, device=device).sum(), op=ReduceOp.SUM)
            global_count = all_reduce(torch.tensor(len(l0s), device=device), op=ReduceOp.SUM)
            avg_l0 = (global_sum / global_count).item()
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
