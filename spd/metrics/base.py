"""Custom Metric base class for distributed metric computation.

This module provides a simplified alternative to torchmetrics.Metric that
supports distributed training with synchronized state across ranks.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import torch
from torch import Tensor

from spd.utils.distributed_utils import gather_all_tensors


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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._state_names: list[str] = []
        self._state_reduce_fns: dict[str, Literal["sum", "cat"]] = {}
        self._dict_state_keys: dict[str, list[str]] = {}  # Maps dict state names to their keys

    def _validate_state_value(
        self, value: Tensor | list[Any], dist_reduce_fx: Literal["sum", "cat"], context: str = ""
    ) -> None:
        match dist_reduce_fx:
            case "sum":
                assert isinstance(value, Tensor), (
                    f"sum reduce requires Tensor{context}, got {type(value)}"
                )
            case "cat":
                assert isinstance(value, list) and len(value) == 0, (
                    f"cat reduce requires empty list{context}, got {type(value)}"
                )

    def add_state(
        self,
        name: str,
        default: Tensor | list[Any] | dict[str, Tensor] | dict[str, list[Any]],
        dist_reduce_fx: Literal["sum", "cat"],
    ) -> None:
        """Register a state variable that should be synchronized across ranks.

        Args:
            name: Name of the state variable
            default: Default value (Tensor for "sum", empty list for "cat", or dict of these)
            dist_reduce_fx: How to reduce across ranks ("sum" or "cat")

        Raises:
            AssertionError: If default value doesn't match reduce function
        """
        assert dist_reduce_fx in ["sum", "cat"], f"Invalid reduce function: {dist_reduce_fx}"

        if isinstance(default, dict):
            self._dict_state_keys[name] = list(default.keys())
            for key, value in default.items():
                self._validate_state_value(value, dist_reduce_fx, context=f" for key '{key}'")
        else:
            self._validate_state_value(default, dist_reduce_fx)

        setattr(self, name, default)
        self._state_names.append(name)
        self._state_reduce_fns[name] = dist_reduce_fx

    def _reduce_state_value(
        self, value: Tensor | list[Tensor], reduce_fn: Literal["sum", "cat"]
    ) -> Tensor | list[Tensor]:
        """Reduce a single state value across distributed ranks.

        Args:
            value: The value to reduce (Tensor for sum, list of Tensors for cat)
            reduce_fn: The reduction function

        Returns:
            The reduced value as a Tensor, or empty list if input was empty list
        """
        assert reduce_fn in ["sum", "cat"], f"Invalid reduce function: {reduce_fn}"
        match reduce_fn:
            case "sum":
                assert isinstance(value, Tensor), "sum reduce requires Tensor value"
                # Note: all_reduce would be more efficient but using gather_all_tensors for
                # consistency and because communication costs for evals shouldn't be large.
                gathered = gather_all_tensors(value)
                return cast(Tensor, sum(gathered))
            case "cat":
                assert isinstance(value, list), "cat reduce requires list value"
                if len(value) == 0:
                    return value

                local_tensor = torch.cat(value, dim=0)
                gathered = gather_all_tensors(local_tensor)
                return torch.cat(gathered, dim=0)

    def sync_dist(self) -> None:
        """Synchronize all registered states across distributed ranks.

        For "sum" reduction: gathers tensors from all ranks and sums them.
        For "cat" reduction: concatenates list elements locally, then gathers
        and concatenates across all ranks.

        For dictionary states: applies the reduction to each value in the dict.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return

        for name in self._state_names:
            state = getattr(self, name)
            reduce_fn = self._state_reduce_fns[name]

            if name in self._dict_state_keys:
                assert isinstance(state, dict), (
                    f"Expected dict for state '{name}', got {type(state)}"
                )
                synced_dict = {
                    key: self._reduce_state_value(state[key], reduce_fn)
                    for key in self._dict_state_keys[name]
                }
                setattr(self, name, synced_dict)
            else:
                reduced = self._reduce_state_value(state, reduce_fn)
                # Skip empty lists for cat reduction
                if not (reduce_fn == "cat" and isinstance(reduced, list)):
                    setattr(self, name, reduced)

    def _move_value_to_device(
        self, value: Tensor | list[Tensor], device: torch.device | str
    ) -> Tensor | list[Tensor]:
        """Move a value to the specified device."""
        assert isinstance(value, Tensor | list), "Value must be Tensor or list"
        match value:
            case Tensor():
                return value.to(device)
            case list():
                items = []
                for item in value:
                    assert isinstance(item, Tensor), "List must contain Tensors"
                    items.append(item.to(device))
                return items

    def to(self, device: torch.device | str) -> "Metric":
        """Move all tensor states to the specified device.

        Args:
            device: Target device (e.g., "cuda", "cpu", or torch.device)

        Returns:
            self for method chaining
        """
        for name in self._state_names:
            state = getattr(self, name)
            if isinstance(state, dict):
                state = {key: self._move_value_to_device(val, device) for key, val in state.items()}
            else:
                state = self._move_value_to_device(state, device)
            setattr(self, name, state)

        return self

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

        For training with DDP: call `compute()` directly to get per-rank metrics
        For evaluation: call `sync_dist()` then `compute()` to get global metrics
        """
        pass
