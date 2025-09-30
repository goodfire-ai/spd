"""Custom Metric base class for distributed metric computation.

This module provides a simplified alternative to torchmetrics.Metric that
supports distributed training with synchronized state across ranks.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor


def _gather_all_tensors(tensor: Tensor, group: Any = None) -> list[Tensor]:
    """Gather tensors from all distributed processes.

    Requires all tensors to have identical shapes across all ranks.

    Args:
        tensor: The tensor to gather from all ranks
        group: The process group (defaults to WORLD)

    Returns:
        List of tensors from all ranks (including local rank)

    Raises:
        AssertionError: If tensors have different shapes across ranks
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [tensor]

    if group is None:
        group = torch.distributed.group.WORLD

    tensor = tensor.contiguous()
    world_size = torch.distributed.get_world_size(group)
    current_rank = torch.distributed.get_rank(group)

    # Gather sizes from all ranks
    local_size = torch.tensor(tensor.shape, device=tensor.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)

    # Assert all shapes are identical
    for rank, size in enumerate(local_sizes):
        assert torch.equal(size, local_size), (
            f"Shape mismatch in distributed gather: "
            f"rank {rank} has shape {size.tolist()}, "
            f"rank {current_rank} (current) has shape {local_size.tolist()}"
        )

    # Gather tensors from all ranks
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor, group=group)

    # Replace our rank's entry with the original to preserve autograd
    gathered[current_rank] = tensor

    return gathered


class Metric(ABC):
    """Base class for metrics with distributed synchronization support.

    This class provides similar functionality to torchmetrics.Metric.

    Subclasses should:
    1. Call `add_state()` in `__init__()` to register metric states
    2. Implement `update()` to accumulate metric values during training
    3. Implement `compute()` to calculate final metric values

    For distributed evaluation:
    - Call `sync_dist()` before `compute()` to synchronize states across ranks
    - This gives you explicit control over when synchronization happens
    """

    is_differentiable: bool | None = None
    slow: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._state_names: list[str] = []
        self._state_reduce_fns: dict[str, Literal["sum", "cat"]] = {}

    def add_state(
        self,
        name: str,
        default: Tensor | list[Any],
        dist_reduce_fx: Literal["sum", "cat"],
    ) -> None:
        """Register a state variable that should be synchronized across ranks.

        Args:
            name: Name of the state variable
            default: Default value (Tensor for "sum", empty list for "cat")
            dist_reduce_fx: How to reduce across ranks ("sum" or "cat")

        Raises:
            AssertionError: If default value doesn't match reduce function
        """
        assert dist_reduce_fx in ["sum", "cat"], f"Invalid reduce function: {dist_reduce_fx}"
        match dist_reduce_fx:
            case "sum":
                assert isinstance(default, Tensor), (
                    f"sum reduce requires Tensor default, got {type(default)}"
                )
            case "cat":
                assert isinstance(default, list) and len(default) == 0, (
                    f"cat reduce requires empty list default, got {type(default)}"
                )

        setattr(self, name, default)
        self._state_names.append(name)
        self._state_reduce_fns[name] = dist_reduce_fx

    def sync_dist(self) -> None:
        """Synchronize all registered states across distributed ranks.

        For "sum" reduction: gathers tensors from all ranks and sums them.
        For "cat" reduction: concatenates list elements locally, then gathers
        and concatenates across all ranks.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return

        for name in self._state_names:
            state = getattr(self, name)
            reduce_fn = self._state_reduce_fns[name]

            if reduce_fn == "sum":
                # Note, this could be quicker with a torch.distributed.all_reduce, but this is
                # simpler and I don't expect the performance to be a bottleneck for evals
                gathered = _gather_all_tensors(state)
                setattr(self, name, sum(gathered))

            elif reduce_fn == "cat":
                if len(state) == 0:
                    continue

                # Concatenate local list elements
                local_tensor = torch.cat(state, dim=0)

                # Gather from all ranks
                gathered = _gather_all_tensors(local_tensor)

                # Concatenate across ranks and replace state
                setattr(self, name, torch.cat(gathered, dim=0))

    def to(self, device: torch.device | str) -> "Metric":
        """Move all tensor states to the specified device.

        Args:
            device: Target device (e.g., "cuda", "cpu", or torch.device)

        Returns:
            self for method chaining
        """
        for name in self._state_names:
            state = getattr(self, name)

            if isinstance(state, Tensor):
                setattr(self, name, state.to(device))
            elif isinstance(state, list):
                # Move all tensors in the list
                moved_list = [
                    item.to(device) if isinstance(item, Tensor) else item for item in state
                ]
                setattr(self, name, moved_list)

        return self

    def reset(self) -> None:
        """Reset the metric state."""
        for name in self._state_names:
            match self._state_reduce_fns[name]:
                case "sum":
                    setattr(self, name, torch.tensor(0.0))
                case "cat":
                    setattr(self, name, [])

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update metric state with a batch of data.

        This method should be called for each batch during training/evaluation.
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Compute the final metric value(s).

        This method operates on the current state, which may be:
        - Local (per-rank) state if called directly after updates
        - Synchronized (all-ranks) state if called after `sync_dist()`

        For training: call `compute()` directly to get per-rank metrics
        For evaluation: call `sync_dist()` then `compute()` to get global metrics
        """
        pass
