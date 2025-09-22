"""Insert identity operations into models, before specified modules.

This works by inserting a Linear layer initialized as the identity matrix, as a property on the module, then adding a
forward pre-hook to the module that multiplies the input by the identity matrix.

This allows downstream functionality to act as if the identity matrix is just a regular part of the model.
"""

from typing import Any, Literal

import torch
import torch.nn as nn
from transformers.modeling_utils import Conv1D as RadfordConv1D

from spd.log import logger
from spd.models.component_model import SUPPORTED_MODULES, ComponentModel
from spd.utils.distributed_utils import is_main_process


def pre_id_hook(
    mod: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[Any, Any],
) -> tuple[tuple[Any, ...], dict[Any, Any]]:
    assert len(args) == 1, f"Expected 1 positional arg, got {len(args)}"
    # assert no kwargs. This may be overkill. can consider passing kwargs through later but this is
    # simple for now.
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

    # Add identity layers and hooks
    for module_path in identity_module_paths:
        module = target_model.get_submodule(module_path)
        assert isinstance(module, SUPPORTED_MODULES), f"Module {module} not supported"

        match module:
            case nn.Linear():
                _, d_in = module.weight.shape
            case RadfordConv1D():
                d_in, _ = module.weight.shape
            case nn.Embedding():
                raise ValueError("Embedding modules not supported for identity insertion")

        # Create identity linear layer
        pre_identity = nn.Linear(d_in, d_in, bias=False, device=device)
        nn.init.eye_(pre_identity.weight)  # Initialize as identity matrix
        module.pre_identity = pre_identity  # type: ignore

        module.register_forward_pre_hook(pre_id_hook, with_kwargs=True)

        if is_main_process():
            logger.info(f"  Added identity layer to {module_path} with dimension {d_in}")
