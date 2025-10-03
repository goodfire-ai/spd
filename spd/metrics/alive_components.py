"""Track which components are alive based on their firing frequency."""

import torch
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.utils.distributed_utils import all_reduce


class AliveComponentsTracker:
    """Track which components are considered alive based on their firing frequency.

    A component is considered alive if it has fired (importance > threshold) within
    the last n_examples_until_dead examples.

    NOTE: This does not directly inherit from spd.metrics.base.Metric, but its update and compute
    methods have a similar signature to the Metric interface.
    """

    def __init__(
        self,
        module_paths: list[str],
        C: int,
        device: str,
        n_examples_until_dead: int,
        ci_alive_threshold: float,
        global_n_examples_per_batch: int,
    ) -> None:
        """Initialize the tracker.

        Args:
            module_paths: List of module names to track
            C: Number of components per module
            device: Device to store tensors on
            n_examples_until_dead: Number of examples without firing before component is considered dead
            ci_alive_threshold: Causal importance threshold above which a component is considered 'firing'
            global_n_examples_per_batch: Number of examples per batch across all ranks (including
                batch and sequence dimensions)
        """
        self.n_examples_until_dead = n_examples_until_dead
        self.ci_alive_threshold = ci_alive_threshold
        self.n_batches_until_dead = self.n_examples_until_dead // global_n_examples_per_batch

        self.n_batches_since_fired: dict[str, Int[Tensor, " C"]] = {
            m: torch.zeros(C, dtype=torch.int64, device=device) for m in module_paths
        }

    def update(self, ci: dict[str, Float[Tensor, "... C"]]) -> None:
        """Update tracking based on importance values from a batch.

        Args:
            ci: Dict mapping module names to causal importance tensors with shape (..., C)
        """
        for module_name, importance_vals in ci.items():
            firing: Bool[Tensor, " C"] = reduce(
                importance_vals > self.ci_alive_threshold, "... C -> C", torch.any
            )
            self.n_batches_since_fired[module_name] = torch.where(
                firing,
                0,
                self.n_batches_since_fired[module_name] + 1,
            )

    def compute(self) -> dict[str, int]:
        """Compute the number of alive components per module.

        Returns:
            Dict mapping module names to number of alive components,
            with keys formatted as "n_alive/{module_name}"
        """
        out: dict[str, int] = {}
        for module_name in self.n_batches_since_fired:
            # Use MIN reduction so that a component is alive if it fired on ANY rank
            batches_since_fired_reduced = all_reduce(
                self.n_batches_since_fired[module_name], op=ReduceOp.MIN
            )
            out[f"n_alive/{module_name}"] = int(
                (batches_since_fired_reduced < self.n_batches_until_dead).sum().item()
            )
        return out
