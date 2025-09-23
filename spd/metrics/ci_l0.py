from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, override

import wandb
from jaxtyping import Float
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.utils.component_utils import calc_ci_l_zero


class CI_L0(Metric):
    slow = False
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: Any,  # unused
        config: Config,
        groups: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.l0_threshold = config.ci_alive_threshold
        self.groups = groups
        self.l0s: dict[str, list[float]] = defaultdict(list)

    @override
    def update(
        self,
        batch: Tensor | None = None,
        target_out: Tensor | None = None,
        ci: dict[str, Float[Tensor, "... C"]] | None = None,
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]] | None = None,
        **kwargs: Any,
    ) -> None:
        import re

        if ci is None:
            return
        group_sums = defaultdict(float) if self.groups else {}
        for layer_name, layer_ci in ci.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            self.l0s[layer_name].append(l0_val)
            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break
        for group_name, group_sum in group_sums.items():
            self.l0s[group_name].append(group_sum)

    @override
    def compute(self) -> Mapping[str, float | Any]:
        out: dict[str, float | Any] = {}
        table_data = []
        for name, l0s in self.l0s.items():
            avg_l0 = sum(l0s) / max(1, len(l0s))
            out[f"l0_{self.l0_threshold}/{name}"] = avg_l0
            table_data.append((name, avg_l0))
        bar_chart = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{self.l0_threshold}",
        )
        out["l0_bar_chart"] = bar_chart
        return out
