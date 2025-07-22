"""Track which components are alive based on their firing frequency."""

import torch
from einops import reduce
from jaxtyping import Bool, Float
from torch import Tensor


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
        device: torch.device,
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
        self.examples_since_fired_C = {
            module_name: torch.zeros(C, dtype=torch.int64, device=device)
            for module_name in module_names
        }
        self.n_examples_until_dead = n_examples_until_dead
        self.ci_alive_threshold = ci_alive_threshold

    def watch_batch(self, importance_vals_dict_BxC: dict[str, Float[Tensor, "... C"]]) -> None:
        """Update tracking based on importance values from a batch.

        Args:
            importance_vals_dict_BxC: Dict mapping module names to importance tensors
                                     with shape (..., C) where ... represents batch dimensions
        """
        assert set(importance_vals_dict_BxC.keys()) == set(self.module_names), (
            "importance_vals_BxC must have the same keys as module_names"
        )
        for module_name, importance_vals_BxC in importance_vals_dict_BxC.items():
            firing_C: Bool[Tensor, " C"] = reduce(
                importance_vals_BxC > self.ci_alive_threshold, "... C -> C", "any"
            )
            self.examples_since_fired_C[module_name][firing_C] = 0

            n_examples = importance_vals_BxC.shape[:-1].numel()
            self.examples_since_fired_C[module_name][~firing_C] += n_examples

    def n_alive(self) -> dict[str, int]:
        """Get the number of alive components per module.

        Returns:
            Dict mapping module names to number of alive components
        """
        return {
            module_name: int(
                (self.examples_since_fired_C[module_name] < self.n_examples_until_dead).sum().item()
            )
            for module_name in self.module_names
        }
