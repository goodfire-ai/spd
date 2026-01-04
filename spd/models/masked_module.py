from __future__ import annotations

from typing import Literal, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.models.components import Components, ComponentsMaskInfo


class MaskedModule(nn.Module):
    """Wraps a frozen base module with trainable SPD Components, without forward hooks.

    This module is designed to replace an `nn.Linear` / `nn.Embedding` / `Conv1D` / `Identity` leaf
    module inside an arbitrary PyTorch model.

    Why this design (vs forward hooks):
        The previous hook-based approach used `register_forward_hook()` with dynamic
        registration/removal via context managers. This was incompatible with `torch.compile()`
        because dynamic hook management breaks graph tracing. This module-patching approach
        keeps the module structure static, making it compatible with compilation.

    Usage:
        `ComponentModel.forward(...)` MUST call `set_runtime_state()` on each MaskedModule
        BEFORE executing the target model's forward pass. This configures:
        - active: whether to use components (True) or pass through to base module (False)
        - mask_info: component mask, routing mask, and optional weight-delta mask
        - cache_type: whether to cache inputs or component activations
        - cache: dictionary to populate with cached values

        After configuring state, the target model is executed normally, and this wrapper
        applies the component replacement logic in its forward() method.

    Note on torch.compile():
        The mutable state pattern (setting attributes before forward) works with torch.compile(),
        but may cause recompilation if the pattern of `active` flags changes frequently across
        forward calls. For best performance, try to maintain consistent activation patterns.
    """

    def __init__(self, *, module_name: str, base: nn.Module, components: Components) -> None:
        super().__init__()
        self.module_name = module_name
        self.base = base
        self.components = components

        # Per-forward runtime state, set by ComponentModel.forward(...) before execution.
        self.active: bool = False
        self.mask_info: ComponentsMaskInfo | None = None
        self.cache_type: Literal["none", "input", "component_acts"] = "none"
        self.cache: dict[str, Tensor] | None = None

    @override
    def __getattr__(self, name: str) -> Tensor | nn.Module:
        """Forward attribute access to base module for nested submodule compatibility.

        This allows code to access nested submodules (e.g., model.linear1.pre_identity)
        even after linear1 has been wrapped with MaskedModule.
        """
        # nn.Module stores _modules in self._modules dict, access it directly to avoid recursion
        _modules = object.__getattribute__(self, "_modules")
        if "base" in _modules:
            base = _modules["base"]
            if hasattr(base, name):
                return getattr(base, name)
        # Fall back to nn.Module's __getattr__ for standard behavior
        return super().__getattr__(name)

    def set_runtime_state(
        self,
        *,
        active: bool,
        mask_info: ComponentsMaskInfo | None,
        cache_type: Literal["none", "input", "component_acts"],
        cache: dict[str, Tensor] | None,
    ) -> None:
        self.active = active
        self.mask_info = mask_info
        self.cache_type = cache_type
        self.cache = cache

    def validate_state(self) -> None:
        """Validate that runtime state is consistent. Call this for debugging.

        Raises:
            AssertionError: If state is inconsistent (e.g., active=True but mask_info=None).
        """
        if self.active:
            assert self.mask_info is not None, (
                f"MaskedModule '{self.module_name}': active=True but mask_info is None. "
                "set_runtime_state() must be called with mask_info when active=True."
            )
        if self.cache_type != "none":
            assert self.cache is not None, (
                f"MaskedModule '{self.module_name}': cache_type='{self.cache_type}' but cache is None. "
                "set_runtime_state() must be called with a cache dict when caching is enabled."
            )

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self.cache_type == "input":
            assert self.cache is not None
            self.cache[self.module_name] = x

        if not self.active:
            return self.base(x)

        assert self.mask_info is not None
        mask_info = self.mask_info

        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = (
            {} if self.cache_type == "component_acts" else None
        )

        components_out = self.components(
            x,
            mask=mask_info.component_mask,
            weight_delta_and_mask=mask_info.weight_delta_and_mask,
            component_acts_cache=component_acts_cache,
        )

        if component_acts_cache is not None:
            assert self.cache is not None
            for k, v in component_acts_cache.items():
                self.cache[f"{self.module_name}_{k}"] = v

        if mask_info.routing_mask == "all":
            return components_out

        base_out = self.base(x)
        routing_mask: Bool[Tensor, ...] = mask_info.routing_mask
        return torch.where(routing_mask[..., None], components_out, base_out)
