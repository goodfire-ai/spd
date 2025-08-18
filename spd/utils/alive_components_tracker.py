"""Track which components are alive based on their firing frequency."""

import torch
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.utils.distributed_utils import all_reduce, is_distributed


class AliveComponentsTracker:
    """Track which components are considered alive based on their firing frequency.

    A component is considered alive if it has fired (importance > threshold) within
    the last n_examples_until_dead examples.
    """

    def __init__(
        self,
        module_names: list[str],
        C: int,
        n_examples_until_dead: int,
        device: str,
        ci_alive_threshold: float,
    ):
        """Initialize the tracker.

        Args:
            module_names: Names of modules to track
            C: Number of components per module
            n_examples_until_dead: Number of examples without firing before component is considered dead
            device: Device to store tensors on
            ci_alive_threshold: Causal importance threshold above which a component is considered 'firing'
        """
        self.module_names = module_names
        self.examples_since_fired: dict[str, Int[Tensor, " C"]] = {
            module_name: torch.zeros(C, dtype=torch.int64, device=device)
            for module_name in module_names
        }
        self.n_examples_until_dead = n_examples_until_dead
        self.ci_alive_threshold = ci_alive_threshold

    def watch_batch(self, importance_vals_dict: dict[str, Float[Tensor, "... C"]]) -> None:
        """Update tracking based on importance values from a batch.


        Args:
            importance_vals_dict: Dict mapping module names to importance tensors
                                  with shape (..., C) where ... represents batch dimensions
        """
        assert set(importance_vals_dict.keys()) == set(self.module_names), (
            "importance_vals_dict must have the same keys as module_names"
        )
        for module_name, importance_vals in importance_vals_dict.items():
            firing: Bool[Tensor, " C"] = reduce(
                importance_vals > self.ci_alive_threshold, "... C -> C", torch.any
            )
            if is_distributed():
                # Check if any ci value is > threshold on any rank.
                firing = all_reduce(firing, op=ReduceOp.MAX)

            n_examples = importance_vals.shape[:-1].numel()

            self.examples_since_fired[module_name] = torch.where(
                firing,
                0,
                self.examples_since_fired[module_name] + n_examples,
            )

    def n_alive(self) -> dict[str, int]:
        """Get the number of alive components per module.

        Returns:
            Dict mapping module names to number of alive components
        """
        return {
            module_name: int(
                (self.examples_since_fired[module_name] < self.n_examples_until_dead).sum().item()
            )
            for module_name in self.module_names
        }
