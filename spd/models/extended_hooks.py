"""Extended hook system for ComponentModel to capture comprehensive network quantities."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from spd.models.components import Components

logger = logging.getLogger(__name__)


class HookCapture(Protocol):
    """Protocol for hook capture functions."""

    def __call__(
        self,
        module: nn.Module,
        input_args: tuple[Any, ...],
        output: Any,
        capture_data: dict[str, Any],
        **kwargs: Any,
    ) -> None: ...


@dataclass
class HookConfig:
    """Configuration for extended hook system."""

    # Core capture options
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_intermediates: bool = False

    # Component-specific options
    capture_component_inner_acts: bool = True
    capture_ci_fn_outputs: bool = True
    capture_causal_importances: bool = True

    # Performance options
    clone_tensors: bool = True
    detach_tensors: bool = True

    # Model-specific options
    model_specific_hooks: dict[str, Any] = field(default_factory=dict)

    # Validation options
    validate_shapes: bool = True
    log_capture_stats: bool = False


@dataclass
class CaptureMetadata:
    """Metadata for captured data."""

    module_name: str
    hook_type: str
    capture_time: float
    tensor_shape: tuple[int, ...]
    tensor_dtype: torch.dtype
    device: torch.device


class ExtendedHookManager:
    """Manages comprehensive hook registration and data collection."""

    def __init__(self, model: Any, config: HookConfig | None = None):
        self.model = model
        self.config = config or HookConfig()
        self.hook_handles: list[RemovableHandle] = []
        self.capture_data: dict[str, Any] = {}
        self.capture_metadata: dict[str, CaptureMetadata] = {}
        self._capture_count = 0

        # Validate model structure
        self._validate_model_structure()

    def _validate_model_structure(self) -> None:
        """Validate that the model has the expected structure."""
        if not hasattr(self.model, "target_model"):
            raise ValueError("Model must have target_model attribute")
        if not hasattr(self.model, "components"):
            raise ValueError("Model must have components attribute")
        if not hasattr(self.model, "ci_fns"):
            raise ValueError("Model must have ci_fns attribute")

    @contextmanager
    def hooks_active(self) -> Generator[ExtendedHookManager, None, None]:
        """Context manager for hook registration and cleanup."""
        try:
            self._register_hooks()
            yield self
        except Exception as e:
            logger.error(f"Error in hook context: {e}")
            raise
        finally:
            self._cleanup_hooks()

    def _register_hooks(self) -> None:
        """Register all configured hooks with proper error handling."""
        try:
            # Register target model hooks
            self._register_target_model_hooks()

            # Register component hooks
            self._register_component_hooks()

            # Register CI function hooks
            self._register_ci_fn_hooks()

            # Register model-specific hooks
            self._register_model_specific_hooks()

            if self.config.log_capture_stats:
                logger.info(f"Registered {len(self.hook_handles)} hooks")

        except Exception as e:
            logger.error(f"Failed to register hooks: {e}")
            self._cleanup_hooks()
            raise

    def _register_target_model_hooks(self) -> None:
        """Register hooks on target model modules."""
        for module_name in self.model.module_paths:
            try:
                target_module = self.model.target_model.get_submodule(module_name)

                # Create hook function with proper closure
                hook_fn = partial(
                    self._target_module_hook,
                    module_name=module_name,
                    capture_data=self.capture_data,
                    config=self.config,
                )

                handle = target_module.register_forward_hook(hook_fn)
                self.hook_handles.append(handle)

            except Exception as e:
                logger.warning(f"Failed to register hook for {module_name}: {e}")

    def _register_component_hooks(self) -> None:
        """Register hooks on component modules."""
        for module_name, component in self.model.components.items():
            try:
                # Hook for component inner activations
                if self.config.capture_component_inner_acts:
                    hook_fn = partial(
                        self._component_inner_acts_hook,
                        module_name=module_name,
                        capture_data=self.capture_data,
                        config=self.config,
                    )
                    handle = component.register_forward_hook(hook_fn)
                    self.hook_handles.append(handle)

            except Exception as e:
                logger.warning(f"Failed to register component hook for {module_name}: {e}")

    def _register_ci_fn_hooks(self) -> None:
        """Register hooks on CI function modules."""
        for module_name, ci_fn in self.model.ci_fns.items():
            try:
                if self.config.capture_ci_fn_outputs:
                    hook_fn = partial(
                        self._ci_fn_hook,
                        module_name=module_name,
                        capture_data=self.capture_data,
                        config=self.config,
                    )
                    handle = ci_fn.register_forward_hook(hook_fn)
                    self.hook_handles.append(handle)

            except Exception as e:
                logger.warning(f"Failed to register CI function hook for {module_name}: {e}")

    def _register_model_specific_hooks(self) -> None:
        """Register model-specific hooks based on target model type."""
        target_model = self.model.target_model

        # ResidMLP specific hooks
        if hasattr(target_model, "layers") and hasattr(target_model, "act_fn"):
            self._register_residmlp_hooks(target_model)

        # Add other model types as needed
        # elif isinstance(target_model, SomeOtherModel):
        #     self._register_other_model_hooks(target_model)

    def _register_residmlp_hooks(self, model: nn.Module) -> None:
        """Register ResidMLP-specific hooks for intermediate activations."""
        if not self.config.capture_intermediates:
            return

        try:
            # Hook for pre-activation values (output of mlp_in linear layer)
            for i, layer in enumerate(model.layers):
                if hasattr(layer, "mlp_in"):
                    hook_fn = partial(
                        self._residmlp_pre_activation_hook,
                        layer_idx=i,
                        capture_data=self.capture_data,
                        config=self.config,
                    )
                    handle = layer.mlp_in.register_forward_hook(hook_fn)
                    self.hook_handles.append(handle)

            # Hook for post-activation values (after ReLU) by hooking the mlp_in output
            # We'll capture the post-activation by hooking the mlp_in layer and applying ReLU
            for i, layer in enumerate(model.layers):
                hook_fn = partial(
                    self._residmlp_post_activation_hook,
                    layer_idx=i,
                    capture_data=self.capture_data,
                    config=self.config,
                    act_fn=model.act_fn,
                )
                handle = layer.mlp_in.register_forward_hook(hook_fn)
                self.hook_handles.append(handle)

        except Exception as e:
            logger.warning(f"Failed to register ResidMLP hooks: {e}")

    def _cleanup_hooks(self) -> None:
        """Safely remove all registered hooks."""
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception as e:
                logger.warning(f"Failed to remove hook: {e}")

        self.hook_handles.clear()

    # Hook functions with proper error handling
    def _target_module_hook(
        self,
        _module: nn.Module,
        input_args: tuple[Any, ...],
        output: Any,
        module_name: str,
        capture_data: dict[str, Any],
        config: HookConfig,
    ) -> None:
        """Hook for target module inputs and outputs."""
        try:
            if config.capture_inputs and input_args:
                x = input_args[0]
                if isinstance(x, Tensor):
                    data = self._prepare_tensor(x, config)
                    capture_data.setdefault("target_inputs", {})[module_name] = data
                    self._record_metadata(module_name, "target_input", data)

            if config.capture_outputs and isinstance(output, Tensor):
                data = self._prepare_tensor(output, config)
                capture_data.setdefault("target_outputs", {})[module_name] = data
                self._record_metadata(module_name, "target_output", data)

        except Exception as e:
            logger.warning(f"Error in target module hook for {module_name}: {e}")

    def _component_inner_acts_hook(
        self,
        module: nn.Module,
        input_args: tuple[Any, ...],
        _output: Any,
        module_name: str,
        capture_data: dict[str, Any],
        config: HookConfig,
    ) -> None:
        """Hook for component inner activations."""
        try:
            if isinstance(module, Components) and input_args:
                x = input_args[0]
                if isinstance(x, Tensor):
                    inner_acts = module.get_inner_acts(x)
                    data = self._prepare_tensor(inner_acts, config)
                    capture_data.setdefault("component_inner_acts", {})[module_name] = data
                    self._record_metadata(module_name, "component_inner_acts", data)

        except Exception as e:
            logger.warning(f"Error in component inner acts hook for {module_name}: {e}")

    def _ci_fn_hook(
        self,
        _module: nn.Module,
        _input_args: tuple[Any, ...],
        output: Any,
        module_name: str,
        capture_data: dict[str, Any],
        config: HookConfig,
    ) -> None:
        """Hook for CI function outputs."""
        try:
            if isinstance(output, Tensor):
                data = self._prepare_tensor(output, config)
                capture_data.setdefault("ci_fn_outputs", {})[module_name] = data
                self._record_metadata(module_name, "ci_fn_output", data)

        except Exception as e:
            logger.warning(f"Error in CI function hook for {module_name}: {e}")

    def _residmlp_pre_activation_hook(
        self,
        _module: nn.Module,
        _input_args: tuple[Any, ...],
        output: Any,
        layer_idx: int,
        capture_data: dict[str, Any],
        config: HookConfig,
    ) -> None:
        """Hook for ResidMLP pre-activation values."""
        try:
            if isinstance(output, Tensor):
                data = self._prepare_tensor(output, config)
                layer_name = f"layers.{layer_idx}.mlp_in"
                capture_data.setdefault("residmlp_pre_activations", {})[layer_name] = data
                self._record_metadata(layer_name, "residmlp_pre_activation", data)

        except Exception as e:
            logger.warning(f"Error in ResidMLP pre-activation hook for layer {layer_idx}: {e}")

    def _residmlp_post_activation_hook(
        self,
        _module: nn.Module,
        _input_args: tuple[Any, ...],
        output: Any,
        layer_idx: int,
        capture_data: dict[str, Any],
        config: HookConfig,
        act_fn: Any,
    ) -> None:
        """Hook for ResidMLP post-activation values (after ReLU)."""
        try:
            if isinstance(output, Tensor):
                # The output is the pre-activation from mlp_in
                pre_act = output

                # Apply activation function to get post-activation
                post_act = act_fn(pre_act)

                if isinstance(post_act, Tensor):
                    data = self._prepare_tensor(post_act, config)
                    layer_name = f"layers.{layer_idx}.mlp_in"
                    capture_data.setdefault("residmlp_post_activations", {})[layer_name] = data
                    self._record_metadata(layer_name, "residmlp_post_activation", data)

        except Exception as e:
            logger.warning(f"Error in ResidMLP post-activation hook for layer {layer_idx}: {e}")

    def _prepare_tensor(self, tensor: Tensor, config: HookConfig) -> Tensor:
        """Prepare tensor for capture based on configuration."""
        if config.detach_tensors:
            tensor = tensor.detach()

        if config.clone_tensors:
            tensor = tensor.clone()

        return tensor

    def _record_metadata(self, module_name: str, hook_type: str, tensor: Tensor) -> None:
        """Record metadata for captured tensor."""
        import time

        self._capture_count += 1
        key = f"{module_name}_{hook_type}_{self._capture_count}"

        self.capture_metadata[key] = CaptureMetadata(
            module_name=module_name,
            hook_type=hook_type,
            capture_time=time.time(),
            tensor_shape=tuple(tensor.shape),
            tensor_dtype=tensor.dtype,
            device=tensor.device,
        )

    def get_capture_summary(self) -> dict[str, Any]:
        """Get summary of captured data."""
        summary = {
            "total_captures": self._capture_count,
            "hook_handles": len(self.hook_handles),
            "data_keys": list(self.capture_data.keys()),
            "metadata_keys": list(self.capture_metadata.keys()),
        }

        # Add shape information
        for key, data in self.capture_data.items():
            if isinstance(data, dict):
                summary[f"{key}_shapes"] = {
                    k: tuple(v.shape) if isinstance(v, Tensor) else str(type(v))
                    for k, v in data.items()
                }

        return summary
