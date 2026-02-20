"""Hook management for semantic backpropagation.

This module provides direct integration of PyTorch hooks for gradient steering,
activation caching, and override mechanisms. Previously wrapped LlamaScope from sbp2,
now contains the full implementation for a self-contained library.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LlamaScope:
    """Class for adding, using, and removing PyTorch hooks with a model.

    This is the core hook management system that enables:
    - Activation caching during forward pass
    - Gradient caching during backward pass
    - Activation overriding for steering
    - Gradient transformation for interventions
    """

    def __init__(self, model: nn.Module):
        """Initialize the scope with a model.

        Args:
            model: The PyTorch model to attach hooks to
        """
        self.model = model
        self.hooks: dict[str, Any] = {}
        self.activations_cache: dict[str, list[torch.Tensor]] = {}
        self.override_store: dict[str, torch.Tensor | None] = {}
        self.indexed_override_store: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
        self.backwards_hooks: dict[str, Any] = {}
        self.gradient_cache: dict[str, list[torch.Tensor]] = {}
        self._build_module_dict()

    # ============ Module listing ============
    def _build_module_dict(self):
        """Walks the model's module tree and builds a name: module map."""
        self._module_dict = {}

        def recurse(module, prefix=''):
            """Recursive tree walk to build self._module_dict."""
            for name, child in module.named_children():
                self._module_dict[prefix+name] = child
                recurse(child, prefix=prefix+name+'-')

        recurse(self.model)  # build the tree

    def list_modules(self):
        """Lists all modules in the module dictionary."""
        return self._module_dict.keys()

    # ============ Generic hook registration ============
    def add_hook(self, hook_fn, module_str: str, hook_name: str):
        """Add a hook_fn to the module given by module_str."""
        module = self._module_dict[module_str]
        hook_handle = module.register_forward_hook(hook_fn)
        self.hooks[hook_name] = hook_handle

    # ============ Activations caching ============
    def _build_caching_hook(self, module_str: str):
        """Build a hook function for caching activations."""
        self.activations_cache[module_str] = []
        def hook_fn(model, input, output):
            self.activations_cache[module_str].append(output)
        return hook_fn

    def add_caching_hook(self, module_str: str):
        """Adds an activations caching hook at the location in module_str."""
        hook_fn = self._build_caching_hook(module_str)
        self.add_hook(hook_fn, module_str, 'cache-'+module_str)

    def clear_cache(self, module_str: str):
        """Clears the activations cache corresponding to module_str."""
        if module_str not in self.activations_cache.keys():
            raise KeyError(f'No activations cache for {module_str}.')
        else:
            self.activations_cache[module_str] = []

    def clear_all_caches(self):
        """Clear all activation caches."""
        for module_str in self.activations_cache.keys():
            self.clear_cache(module_str)

    def remove_cache(self, module_str: str):
        """Remove the cache for module_str."""
        del self.activations_cache[module_str]

    def remove_all_caches(self):
        """Remove all caches."""
        caches = list(self.activations_cache.keys())
        for cache_str in caches:
            self.remove_cache(cache_str)

    # ============ Activation override ============
    def _build_override_hook(self, module_str: str):
        """Build a hook function for overriding activations."""
        self.override_store[module_str] = None  # won't override when None
        def hook_fn(model, input, output):
            if self.override_store[module_str] is not None:
                return output + self.override_store[module_str].to(output.device)
            else:
                return output
        return hook_fn

    def add_override_hook(self, module_str: str):
        """Adds hook to override output of module_str using override_store."""
        hook_fn = self._build_override_hook(module_str)
        self.add_hook(hook_fn, module_str, 'override-'+module_str)

    def override(self, module_str: str, override_tensor: torch.Tensor):
        """Sets the override tensor for module_str."""
        self.override_store[module_str] = override_tensor

    def clear_override(self, module_str: str):
        """Clear override hook so it won't affect forward pass."""
        self.override_store[module_str] = None

    def clear_all_overrides(self):
        """Clear all override hooks."""
        overrides = list(self.override_store.keys())
        for override in overrides:
            self.clear_override(override)

    # ============ Indexed activation override ============
    def _build_indexed_override_hook(self, module_str: str):
        """Build a hook function for overriding activations at specific token indices.

        The indexed override allows steering only at specific positions in the sequence,
        rather than uniformly across all positions.

        Supports modules that return tuples (e.g., MultiheadAttention).
        """
        self.indexed_override_store[module_str] = (None, None)  # (override_tensor, indices)
        def hook_fn(model, input, output):
            override_tensor, indices = self.indexed_override_store[module_str]
            if override_tensor is not None and indices is not None:
                # Handle tuple outputs (e.g., from attention layers)
                if isinstance(output, tuple):
                    base_output = output[0]
                    other_outputs = output[1:]
                else:
                    base_output = output
                    other_outputs = None

                # Create a mask of zeros with the same shape as base output
                steering = torch.zeros_like(base_output)

                # Apply the override only at the specified indices
                # indices shape: (batch_size,) or (batch_size, num_indices)
                # override_tensor shape: (batch_size, d_model) or (batch_size, num_indices, d_model)
                if indices.dim() == 1:
                    # Single index per batch item: (batch_size,)
                    batch_indices = torch.arange(base_output.shape[0], device=base_output.device)
                    steering[batch_indices, indices] = override_tensor.to(base_output.device)
                else:
                    # Multiple indices per batch item: (batch_size, num_indices)
                    batch_size, num_indices = indices.shape
                    for b in range(batch_size):
                        for i, idx in enumerate(indices[b]):
                            if override_tensor.dim() == 2:
                                # Same vector for all indices: (batch_size, d_model)
                                steering[b, idx] = override_tensor[b].to(base_output.device)
                            else:
                                # Different vector per index: (batch_size, num_indices, d_model)
                                steering[b, idx] = override_tensor[b, i].to(base_output.device)

                steered_output = base_output + steering
                if other_outputs is not None:
                    return (steered_output,) + other_outputs
                else:
                    return steered_output
            else:
                return output
        return hook_fn

    def add_indexed_override_hook(self, module_str: str):
        """Adds hook to override output of module_str at specific token indices."""
        hook_fn = self._build_indexed_override_hook(module_str)
        self.add_hook(hook_fn, module_str, 'indexed-override-'+module_str)

    def indexed_override(self, module_str: str, override_tensor: torch.Tensor, indices: torch.Tensor):
        """Sets the indexed override tensor and token indices for module_str.

        Args:
            module_str: The module to apply the override to
            override_tensor: Steering vector(s) to apply
                Shape options:
                - (batch_size, d_model): Same vector applied to all specified indices
                - (batch_size, num_indices, d_model): Different vector per index
            indices: Token positions to apply steering
                Shape options:
                - (batch_size,): Single index per batch item
                - (batch_size, num_indices): Multiple indices per batch item

        Raises:
            ValueError: If batch sizes don't match between override_tensor and indices
        """
        # Validate batch sizes match
        override_batch_size = override_tensor.shape[0]
        indices_batch_size = indices.shape[0]

        if override_batch_size != indices_batch_size:
            raise ValueError(
                f"Batch size mismatch: override_tensor has batch_size={override_batch_size} "
                f"but indices has batch_size={indices_batch_size}. "
                f"Both must have the same batch dimension."
            )

        self.indexed_override_store[module_str] = (override_tensor, indices)

    def clear_indexed_override(self, module_str: str):
        """Clear indexed override hook so it won't affect forward pass."""
        self.indexed_override_store[module_str] = (None, None)

    def clear_all_indexed_overrides(self):
        """Clear all indexed override hooks."""
        overrides = list(self.indexed_override_store.keys())
        for override in overrides:
            self.clear_indexed_override(override)

    # ============ Backwards hooks ============
    def add_backward_hook(self, hook_fn, module_str: str, hook_name: str):
        """Add a backward hook_fn to the module given by module_str."""
        module = self._module_dict[module_str]
        hook_handle = module.register_full_backward_hook(hook_fn)
        self.backwards_hooks[hook_name] = hook_handle

    def add_backward_pre_hook(self, hook_fn, module_str: str, hook_name: str):
        """Add a backward pre-hook_fn to the module given by module_str."""
        module = self._module_dict[module_str]
        hook_handle = module.register_full_backward_pre_hook(hook_fn)
        self.backwards_hooks[hook_name] = hook_handle

    def _build_gradient_hook(self, module_str: str):
        """Build a hook function for caching gradients."""
        self.gradient_cache[module_str] = []
        def hook_fn(module, grad_input, grad_output):
            self.gradient_cache[module_str].append(grad_output)
        return hook_fn

    def add_gradient_hook(self, module_str: str):
        """Add a gradient caching hook to the module given by module_str."""
        hook_fn = self._build_gradient_hook(module_str)
        self.add_backward_hook(hook_fn, module_str, 'gradient-'+module_str)

    def _build_gradient_transform_hook(self, module_str: str, matrix: torch.Tensor):
        """Build a hook function for transforming gradients with a matrix."""
        def hook_fn(module, grad_output):
            retval = (grad_output[0] @ matrix).unsqueeze(0)
            return retval
        return hook_fn

    def add_gradient_transform_hook(self, module_str: str, matrix: torch.Tensor):
        """Add a gradient transformation hook to the module."""
        hook_fn = self._build_gradient_transform_hook(module_str, matrix)
        self.add_backward_pre_hook(hook_fn, module_str, 'gradient-transform-'+module_str)

    def clear_gradient(self, module_str: str):
        """Clear the gradient cache for the module given by module_str."""
        if module_str not in self.gradient_cache.keys():
            raise KeyError(f'No gradient cache for {module_str}.')
        else:
            self.gradient_cache[module_str] = []

    def clear_all_gradients(self):
        """Clear all gradient caches."""
        for module_str in self.gradient_cache.keys():
            self.clear_gradient(module_str)

    # ============ Hook cleanup ============
    def remove_hook(self, hook_name: str):
        """Remove a hook with name hook_name from the model."""
        self.hooks[hook_name].remove()
        del self.hooks[hook_name]

    def remove_backward_hook(self, hook_name: str):
        """Remove a backward hook with name hook_name from the model."""
        self.backwards_hooks[hook_name].remove()
        del self.backwards_hooks[hook_name]

    def remove_all_hooks(self):
        """Remove all hooks from the model."""
        hooks = list(self.hooks.keys())
        for hook_name in hooks:
            self.remove_hook(hook_name)

        bwd_hooks = list(self.backwards_hooks.keys())
        for bwd_hook in bwd_hooks:
            self.remove_backward_hook(bwd_hook)


class HookManager:
    """High-level interface for managing hooks in semantic backpropagation.

    This provides a simplified API on top of LlamaScope for the specific
    operations needed in semantic backpropagation training.
    """

    def __init__(self, model: nn.Module, location: str):
        """Initialize the hook manager.

        Args:
            model: The model to attach hooks to (expects a transformer with .model attribute)
            location: The layer location string (e.g., "layers-19")
        """
        self.model = model
        self.location = location

        # Initialize LlamaScope with the model's inner layers
        # For LLaMA models, the actual layers are in model.model
        if hasattr(model, 'model'):
            self.scope = LlamaScope(model.model)
        else:
            self.scope = LlamaScope(model)

        self._ensure_hooks()

    def _ensure_hooks(self):
        """Set up the necessary hooks for semantic backpropagation."""
        # Always enable these for semantic backpropagation
        self.scope.add_caching_hook(self.location)
        self.scope.add_gradient_hook(self.location)
        self.scope.add_override_hook(self.location)
        self.scope.add_indexed_override_hook(self.location)

    def set_override(self, override_vec: torch.Tensor | None):
        """Set or clear the override vector for steering.

        Args:
            override_vec: The steering vector to add to activations, or None to disable.
                Always expects batched shape: (batch_size, d_model)
                For single examples, use shape (1, d_model).
                Broadcasts correctly to (batch_size, seq_len, d_model) output
        """
        if override_vec is None:
            self.scope.clear_override(self.location)
        else:
            # Always expect batched format: (batch_size, d_model)
            if override_vec.dim() == 2:
                # Batched: (batch_size, d_model) -> (batch_size, 1, d_model) for broadcasting
                self.scope.override(self.location, override_vec.unsqueeze(1))
            elif override_vec.dim() == 1:
                # Single unbatched vector (backward compat): (d_model,) -> (1, 1, d_model)
                self.scope.override(self.location, override_vec.unsqueeze(0).unsqueeze(0))
            else:
                raise ValueError(f"Unexpected override vector shape: {override_vec.shape}. Expected (batch_size, d_model) or (d_model,)")

    def clear_state(self):
        """Clear all cached activations and gradients."""
        self.scope.clear_all_caches()
        self.scope.clear_all_gradients()

    def get_grad_tensor(self) -> torch.Tensor:
        """Get the cached gradient tensor.

        Returns:
            The gradient tensor from the backward pass.
            Always returns shape: (batch, seq, d_model)
        """
        # Match access pattern used in the original code
        grad = self.scope.gradient_cache[self.location][0][0]

        # Always return batched format: (batch, seq, d_model)
        if grad.dim() == 4:
            # Shape is (1, batch, seq, d_model) - remove leading dim
            return grad.squeeze(0)  # -> (batch, seq, d_model)
        elif grad.dim() == 3:
            # Already (batch, seq, d_model)
            return grad
        elif grad.dim() == 2:
            # Single unbatched example (seq, d_model) - add batch dimension
            return grad.unsqueeze(0)  # -> (1, seq, d_model)
        else:
            raise ValueError(f"Unexpected gradient shape: {grad.shape}")

    def get_last_activations(self) -> torch.Tensor:
        """Get the last cached activation tensor.

        Returns:
            The activation tensor from the forward pass
        """
        return self.scope.activations_cache[self.location][0]

    def add_gradient_transform(self, matrix: torch.Tensor):
        """Add a gradient transformation hook.

        Args:
            matrix: The transformation matrix to apply to gradients
        """
        self.scope.add_gradient_transform_hook(self.location, matrix)

    def set_indexed_override(self, override_vec: torch.Tensor | None, indices: torch.Tensor | None):
        """Set or clear the indexed override vector for token-specific steering.

        Args:
            override_vec: The steering vector(s) to add to activations at specific positions, or None to disable.
                Shape options:
                - (batch_size, d_model): Same vector applied to all specified indices
                - (batch_size, num_indices, d_model): Different vector per index
            indices: Token positions to apply steering, or None to disable.
                Shape options:
                - (batch_size,): Single index per batch item
                - (batch_size, num_indices): Multiple indices per batch item
        """
        if override_vec is None or indices is None:
            self.scope.clear_indexed_override(self.location)
        else:
            self.scope.indexed_override(self.location, override_vec, indices)

    def cleanup(self):
        """Remove all hooks from the model."""
        self.scope.remove_all_hooks()
        self.scope.clear_all_caches()
        self.scope.clear_all_gradients()
        self.scope.clear_all_overrides()
        self.scope.clear_all_indexed_overrides()