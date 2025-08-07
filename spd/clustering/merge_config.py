from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)

from spd.spd_types import Probability

MergeConfigKey = Literal[
    "activation_threshold",
    "alpha",
    "iters",
    "check_threshold",
    "pop_component_prob",
    "filter_dead_threshold",
    "rank_cost_fn_name",
]


class MergeConfig(BaseModel):
    activation_threshold: Probability | None = Field(
        default=0.01,
        description="Threshold for considering a component active in a group. If None, use raw scalar causal importances",
    )
    alpha: float = Field(
        default=1.0,
        description="rank weight factor. Higher values mean a higher penalty on 'sending' the component weights",
    )
    iters: PositiveInt = Field(
        default=100,
        description="max number of iterations to run the merge algorithm for.",
    )
    check_threshold: Probability = Field(
        default=0.05,
        description="threshold for considering merge pairs, as a fraction of the range of non-diagonal costs. If 0, always select the pair with the lowest cost. if 1, choose randomly among all pairs.",
    )
    pop_component_prob: Probability = Field(
        default=0,
        description="Probability of popping a component in each iteration. If 0, no components are popped.",
    )
    filter_dead_threshold: float = Field(
        default=0.001,
        description="Threshold for filtering out dead components. If a component's activation is below this threshold, it is considered dead and not included in the merge.",
    )

    # rank_cost_fn: Callable[[float], float] = lambda _: 1.0
    rank_cost_fn_name: str = Field(
        default="const_1",
        description="Name of the rank cost function to use. Options: 'const_1', 'const_2', 'log', 'sqrt'.",
    )

    @property
    def rank_cost_fn(self) -> Callable[[float], float]:
        """Get the rank cost function based on the name."""
        if self.rank_cost_fn_name.startswith("const_"):
            const_value: float = float(self.rank_cost_fn_name.split("_")[1])
            return lambda _: const_value
        elif self.rank_cost_fn_name == "log":
            return lambda x: math.log(x + 1e-8)
        elif self.rank_cost_fn_name == "linear":
            return lambda x: x
        else:
            raise ValueError(
                f"Unknown rank cost function: {self.rank_cost_fn_name}. "
                "Options: 'const_{{value}}' where {{value}} is a float, 'log', 'linear'."
            )

    @property
    def rank_cost_name(self) -> str:
        """Get the name of the rank cost function."""
        return getattr(self.rank_cost_fn, "__name__", str(self.rank_cost_fn))

    @property
    def stable_hash(self) -> str:
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]
