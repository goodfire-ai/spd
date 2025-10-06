"""Metric interface for distributed metric computation.

All metrics implement Metric and typically handle distributed synchronization directly in their
compute() methods.
"""

from typing import Any, ClassVar, Protocol

from jaxtyping import Float, Int
from torch import Tensor


class Metric(Protocol):
    """Interface for metrics that can be used in training and/or evaluation."""

    slow: ClassVar[bool] = False

    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: dict[str, Float[Tensor, "... C"]],
        current_frac_of_training: float,
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        """Update metric state with a batch of data."""
        ...

    def compute(self) -> Any:
        """Compute the final metric value(s)."""
        ...
