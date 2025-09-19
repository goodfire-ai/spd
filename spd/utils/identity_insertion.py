"""Insert identity operations into models, before specified modules.

This works by inserting a Linear layer initialized as the identity matrix, as a property on the module, then adding a
forward pre-hook to the module that multiplies the input by the identity matrix.

This allows downstream functionality to act as if the identity matrix is just a regular part of the model.
"""

from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import is_main_process


def _get_input_sizes(
    target_model: nn.Module,
    identity_module_paths: list[str],
    dummy_input: torch.Tensor,
) -> dict[str, int]:
    cache: dict[str, torch.Tensor] = {}
    handles: list[RemovableHandle] = []

    def cache_hook(_: nn.Module, input: tuple[torch.Tensor, ...], param_name: str) -> None:
        cache[param_name] = input[0]

    for module_name in identity_module_paths:
        module = target_model.get_submodule(module_name)
        handle = module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
        handles.append(handle)

    with torch.no_grad():
        target_model(dummy_input)

    for handle in handles:
        handle.remove()

    out = {}
    for module_name in identity_module_paths:
        assert module_name in cache, f"Module {module_name} not in cache"
        assert cache[module_name].ndim == 3, (
            f"Expected 3D input (batch, seq, d_in), got {cache[module_name].ndim}D"
        )
        out[module_name] = cache[module_name].shape[2]

    return out


# Create hook function with proper closure
def pre_id_hook(
    mod: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[Any, Any],
) -> tuple[tuple[Any, ...], dict[Any, Any]]:
    assert len(args) == 1, f"Expected 1 positional arg, got {len(args)}"
    assert not kwargs, f"Expected no kwargs, got {kwargs.keys()}"
    assert hasattr(mod, "pre_identity"), f"Module {mod} has no pre_identity attribute"
    assert isinstance(mod.pre_identity, nn.Linear), (
        f"Module {mod} pre_identity is not a Linear layer"
    )
    return (mod.pre_identity(args[0]),), {}


InputType = Literal["tokens"] | tuple[Literal["vector"], int]
"""'tokens' implies (batch, seq) of integer tokens. ('vector', d_in) implies (batch, d_in) of floats."""


def insert_identity_operations_(
    target_model: nn.Module,
    identity_patterns: list[str],
    input_type: InputType,
    device: torch.device | str,
) -> None:
    """Insert identity linear layers before specified modules.

    Args:
        target_model: The model to modify
        identity_patterns: Patterns matching modules to prepend identity ops to
        device: Device to place tensors on
    """

    if is_main_process():
        logger.info(f"Inserting identity operations before {len(identity_patterns)} modules")

    identity_module_paths = ComponentModel._get_target_module_paths(target_model, identity_patterns)

    match input_type:
        case "tokens":
            dummy_input = torch.zeros(1, 1, device=device, dtype=torch.long)
        case ("vector", d_in):
            dummy_input = torch.randn(1, d_in, device=device)

    layer_input_sizes = _get_input_sizes(target_model, identity_module_paths, dummy_input)

    # Add identity layers and hooks
    for module_path, d_in in layer_input_sizes.items():
        module = target_model.get_submodule(module_path)

        # Create identity linear layer
        pre_identity = nn.Linear(d_in, d_in, bias=False, device=device)
        nn.init.eye_(pre_identity.weight)  # Initialize as identity matrix
        module.pre_identity = pre_identity  # type: ignore

        module.register_forward_pre_hook(pre_id_hook, with_kwargs=True)

        if is_main_process():
            logger.info(f"  Added identity layer to {module_path} with dimension {d_in}")
