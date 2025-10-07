"""Insert identity operations into models, before specified modules.

This works by inserting an Identity layer, as a property on the module, then adding a
forward pre-hook to the module that calls it before the forward pass.

This allows downstream functionality to act as if the identity operation is just a regular part of
the model, namely, allowing us to decompose the identity operation.
"""

from typing import Any

import torch.nn as nn

# see https://github.com/goodfire-ai/spd/issues/139
from transformers.modeling_utils import (
    Conv1D as RadfordConv1D,  # pyright: ignore[reportAttributeAccessIssue]
)

from spd.log import logger
from spd.models.components import Identity
from spd.utils.distributed_utils import is_main_process
from spd.utils.module_utils import get_target_module_paths


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
    assert isinstance(mod.pre_identity, Identity), (
        f"Module {mod} pre_identity is not an Identity layer"
    )
    return (mod.pre_identity(args[0]),), {}


def insert_identity_operations_(target_model: nn.Module, identity_patterns: list[str]) -> None:
    """Insert identity layers before specified modules.

    Args:
        target_model: The model to modify
        identity_patterns: Patterns matching modules to prepend identity ops to
    """

    if is_main_process():
        logger.info(f"Inserting identity operations before {len(identity_patterns)} modules")

    identity_module_paths = get_target_module_paths(target_model, identity_patterns)

    # Add identity layers and hooks
    for module_path in identity_module_paths:
        module = target_model.get_submodule(module_path)

        match module:
            case nn.Linear():
                _, d_in = module.weight.shape
            case RadfordConv1D():
                d_in, _ = module.weight.shape
            case nn.Embedding():
                raise ValueError("Embedding modules not supported for identity insertion")
            case _:
                raise ValueError(f"Module {module} not supported. type: {type(module)}")

        module.pre_identity = Identity(d_in)  # type: ignore
        module.register_forward_pre_hook(pre_id_hook, with_kwargs=True)

        if is_main_process():
            logger.info(f"  Added identity layer to {module_path} with dimension {d_in}")
