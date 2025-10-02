"""Metric interface for distributed metric computation.

All metrics implement MetricInterface and handle distributed synchronization
directly in their compute() methods using all_reduce() or gather_all_tensors().
"""

from abc import ABC, abstractmethod
from typing import Any

from jaxtyping import Float, Int
from torch import Tensor


class Metric(ABC):
    """Interface for metrics that can be used in training and evaluation."""

    slow: bool = False

    @abstractmethod
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        current_frac_of_training: float,
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        """Update metric state with a batch of data."""
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Compute the final metric value(s)."""
        pass
