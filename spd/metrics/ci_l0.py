import re
from collections import defaultdict
from typing import Any, override

import wandb
from jaxtyping import Float
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_ci_l_zero


class CI_L0(Metric):
    """L0 metric for CI values.

    NOTE: Assumes all batches and sequences are the same size.
    """

    slow = False
    is_differentiable: bool | None = False
    full_state_update: bool | None = False  # Avoid double update calls

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        groups: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.l0_threshold = config.ci_alive_threshold
        self.groups = groups

        # Avoid using e.g. "layers.*.mlp_in" as an attribute
        self.key_to_sanitized: dict[str, str] = {}
        self.sanitized_to_key: dict[str, str] = {}

        all_keys = model.module_paths
        if groups:
            all_keys += list(groups.keys())

        for key in all_keys:
            sanitized_key = key.replace(".", "-").replace("*", "all")
            self.key_to_sanitized[key] = sanitized_key
            self.sanitized_to_key[sanitized_key] = key
            self.add_state(sanitized_key, default=[], dist_reduce_fx="cat")

    @override
    def update(self, *, ci: dict[str, Float[Tensor, "... C"]], **kwargs: Any) -> None:
        group_sums = defaultdict(float) if self.groups else {}
        for layer_name, layer_ci in ci.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            sanitized_key = self.key_to_sanitized[layer_name]
            getattr(self, sanitized_key).append(l0_val)

            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break
        for group_name, group_sum in group_sums.items():
            sanitized_key = self.key_to_sanitized[group_name]
            getattr(self, sanitized_key).append(group_sum)

    @override
    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        out = {}
        table_data = []
        for sanitized_key, key in self.sanitized_to_key.items():
            l0s = getattr(self, sanitized_key)
            avg_l0 = sum(l0s) / len(l0s)
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
