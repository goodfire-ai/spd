"""Track which components are alive based on their firing frequency."""

from typing import Any, override

import torch
import torch.distributed as dist
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.metrics.base import Metric


class AliveComponentsTracker(Metric):
    """Track which components are considered alive based on their firing frequency.

    A component is considered alive if it has fired (importance > threshold) within
    the last n_examples_until_dead examples.
    """

    is_differentiable: bool | None = False

    n_batches_since_fired: dict[str, Int[Tensor, " C"]]
    n_batches_until_dead: int

    def __init__(
        self,
        module_paths: list[str],
        C: int,
        n_examples_until_dead: int,
        ci_alive_threshold: float,
        **kwargs: Any,
    ) -> None:
        """Initialize the tracker.

        Args:
            module_paths: List of module names to track
            C: Number of components per module
            n_examples_until_dead: Number of examples without firing before component is considered dead
            ci_alive_threshold: Causal importance threshold above which a component is considered 'firing'
        """
        super().__init__(**kwargs)
        self.n_examples_until_dead = n_examples_until_dead
        self.ci_alive_threshold = ci_alive_threshold

        self.n_batches_until_dead = -1

        self.add_state(
            "n_batches_since_fired",
            default={m: torch.zeros(C, dtype=torch.int64) for m in module_paths},
            dist_reduce_fx="min",
        )

    @override
    def update(self, *, ci: dict[str, Float[Tensor, "... C"]], **_: Any) -> None:
        """Update tracking based on importance values from a batch.

        Args:
            ci: Dict mapping module names to causal importance tensors with shape (..., C)
        """
        if self.n_batches_until_dead == -1:
            # Get shape from any CI tensor (they all have the same batch dimensions)
            local_n_examples = next(iter(ci.values())).shape[:-1].numel()  # All dims except C
            # Get world size directly from torch.distributed to avoid needing init_distributed()
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            global_n_examples = local_n_examples * world_size
            self.n_batches_until_dead = self.n_examples_until_dead // global_n_examples

        for module_name, importance_vals in ci.items():
            firing: Bool[Tensor, " C"] = reduce(
                importance_vals > self.ci_alive_threshold, "... C -> C", torch.any
            )
            self.n_batches_since_fired[module_name] = torch.where(
                firing,
                0,
                self.n_batches_since_fired[module_name] + 1,
            )

    @override
    def compute(self) -> dict[str, int]:
        """Compute the number of alive components per module.

        Should be called after sync_dist() in distributed settings to count firing across all ranks.

        Returns:
            Dict mapping module names to number of alive components,
            with keys formatted as "n_alive/{module_name}"
        """
        out: dict[str, int] = {}
        for module_name in self.n_batches_since_fired:
            out[f"n_alive/{module_name}"] = int(
                (self.n_batches_since_fired[module_name] < self.n_batches_until_dead).sum().item()
            )
        return out
