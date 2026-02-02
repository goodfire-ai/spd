"""Loaders for LM ComponentModels."""

from collections.abc import Generator, Iterator
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from spd.configs import Config
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.spd_types import ModelPath
from spd.utils.general_utils import resolve_class
from spd.utils.module_utils import expand_module_patterns


class LogitsOnlyWrapper(nn.Module):
    """Wrapper that extracts logits from models that return (logits, loss) tuples."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        out = self.model(*args, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out

    @override
    def get_submodule(self, target: str) -> nn.Module:
        # Delegate to wrapped model so paths don't need "model." prefix
        return self.model.get_submodule(target)

    @override
    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        # Delegate to wrapped model so parameter names don't have "model." prefix
        return self.model.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    @override
    def named_modules(
        self, memo: set[nn.Module] | None = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Generator[tuple[str, nn.Module]]:
        # Delegate to wrapped model so module names don't have "model." prefix
        yield from self.model.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    @override
    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the wrapped model for things like wte, lm_head, etc.
        if name == "model":
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def load_lm_component_model_from_run_info(
    run_info: RunInfo[Config],
) -> ComponentModel[Tensor, Tensor]:
    """Load a trained LM ComponentModel from a run info object."""
    config = run_info.config

    model_class = resolve_class(config.pretrained_model_class)
    if config.pretrained_model_name is not None:
        assert hasattr(model_class, "from_pretrained")
        target_model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        assert issubclass(model_class, LoadableModule)
        assert config.pretrained_model_path is not None
        target_model = model_class.from_pretrained(config.pretrained_model_path)

    target_model.eval()
    target_model.requires_grad_(False)

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    # Wrap the model to extract logits from (logits, loss) tuple outputs
    wrapped_model = LogitsOnlyWrapper(target_model)

    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    comp_model: ComponentModel[Tensor, Tensor] = ComponentModel(
        target_model=wrapped_model,
        module_path_info=module_path_info,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        ci_fn_type=config.ci_fn_type,
        sigmoid_type=config.sigmoid_type,
    )

    comp_model_weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
    handle_deprecated_state_dict_keys_(comp_model_weights)
    comp_model.load_state_dict(comp_model_weights)

    return comp_model


def load_lm_component_model(path: ModelPath) -> ComponentModel[Tensor, Tensor]:
    """Load a trained LM ComponentModel from a wandb or local path."""
    run_info = SPDRunInfo.from_path(path)
    return load_lm_component_model_from_run_info(run_info)
