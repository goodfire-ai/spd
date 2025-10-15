"""Metric interface for distributed metric computation.

All metrics implement Metric and typically handle distributed synchronization directly in their
compute() methods.
"""

from typing import Any, ClassVar, Protocol

from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import CIOutputs


class Metric(Protocol):
    """Interface for metrics that can be used in training and/or evaluation."""

    slow: ClassVar[bool] = False
    metric_section: ClassVar[str]

    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        current_frac_of_training: float,
        weight_deltas: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        """Update metric state with a batch of data."""
        ...

    def compute(self) -> Any:
        """Compute the final metric value(s)."""
        ...
