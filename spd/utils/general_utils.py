import copy
import importlib
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from jaxtyping import Float
from pydantic import BaseModel, PositiveFloat
from pydantic.v1.utils import deep_update
from torch import Tensor

from spd.log import logger
from spd.spd_types import ModelPath

# Avoid seaborn package installation (sns.color_palette("colorblind").as_hex())
COLOR_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def get_device() -> str:
    # NOTE: MPS returns NaNs on TMS when run. Avoiding for now.
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def generate_sweep_id() -> str:
    """Generate a unique sweep ID based on timestamp."""
    return f"sweep_id-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_config[T: BaseModel](
    config_path_or_obj: Path | str | dict[str, Any] | T, config_model: type[T]
) -> T:
    """Load the config of class `config_model`, from various sources.

    Args:
        config_path_or_obj (Union[Path, str, dict, `config_model`]): Can be:
            - config object: must be instance of `config_model`
            - dict: config dictionary
            - str starting with 'json:': JSON string with prefix
            - other str: treated as path to a .yaml file
            - Path: path to a .yaml file
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, dict):
        return config_model(**config_path_or_obj)

    if isinstance(config_path_or_obj, str):
        # Check if it's a prefixed JSON string
        if config_path_or_obj.startswith("json:"):
            config_dict = json.loads(config_path_or_obj[5:])
            return config_model(**config_dict)
        else:
            # Treat as file path
            config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(config_path_or_obj, Path), (
        f"passed config is of invalid type {type(config_path_or_obj)}"
    )
    assert config_path_or_obj.suffix == ".yaml", (
        f"Config file {config_path_or_obj} must be a YAML file."
    )
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


def replace_pydantic_model[BaseModelType: BaseModel](
    model: BaseModelType, *updates: dict[str, Any]
) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


def compute_feature_importances(
    batch_size: int,
    n_features: int,
    importance_val: float | None,
    device: str,
) -> Float[Tensor, "batch_size n_features"]:
    # Defines a tensor where the i^th feature has importance importance^i
    if importance_val is None or importance_val == 1.0:
        importance_tensor = torch.ones(batch_size, n_features, device=device)
    else:
        powers = torch.arange(n_features, device=device)
        importances = torch.pow(importance_val, powers)
        importance_tensor = einops.repeat(
            importances, "n_features -> batch_size n_features", batch_size=batch_size
        )
    return importance_tensor


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    """Get a function that returns the learning rate at a given step.

    Args:
        lr_schedule: The learning rate schedule to use
        lr_exponential_halflife: The halflife of the exponential learning rate schedule
    """
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    else:
        # Exponential
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


def replace_deprecated_param_names(
    params: dict[str, Float[Tensor, "..."]], name_map: dict[str, str]
) -> dict[str, Float[Tensor, "..."]]:
    """Replace old parameter names with new parameter names in a dictionary.

    Args:
        params: The dictionary of parameters to fix
        name_map: A dictionary mapping old parameter names to new parameter names
    """
    for k in list(params.keys()):
        for old_name, new_name in name_map.items():
            if old_name in k:
                params[k.replace(old_name, new_name)] = params[k]
                del params[k]
    return params


def resolve_class(path: str) -> type[nn.Module]:
    """Load a class from a string indicating its import path.

    Args:
        path: The path to the class, e.g. "transformers.LlamaForCausalLM" or
            "spd.experiments.resid_mlp.models.ResidMLP"
    """
    module_path, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_pretrained(
    path_to_class: str,
    model_path: ModelPath | None = None,
    model_name_hf: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Load a model from a path to the class and a model name or path.

    Loads from either huggingface (if model_name_hf is provided) or from a wandb str or local path
    (if model_path is provided).

    Args:
        path_to_class: The path to the class, e.g. "transformers.LlamaForCausalLM" or
            "spd.experiments.resid_mlp.models.ResidMLP"
        model_path: The path to the model, e.g. "wandb:spd/runs/zas5yjdl" or /path/to/checkpoint"
        model_name_hf: The name of the model in the Hugging Face model hub,
            e.g. "SimpleStories/SimpleStories-1.25M"
    """
    assert model_path is not None or model_name_hf is not None, (
        "Either model_path or model_name_hf must be provided."
    )
    model_cls = resolve_class(path_to_class)
    if not hasattr(model_cls, "from_pretrained"):
        raise TypeError(f"{model_cls} lacks a `from_pretrained` method.")
    return model_cls.from_pretrained(model_path or model_name_hf, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]


def extract_batch_data(
    batch_item: dict[str, Any] | tuple[Tensor, ...] | Tensor,
    input_key: str = "input_ids",
) -> Tensor:
    """Extract input data from various batch formats.

    This utility function handles different batch formats commonly used across the codebase:
    1. Dictionary format: {"input_ids": tensor, ...} - common in LM tasks
    2. Tuple format: (input_tensor, labels) - common in SPD optimization
    3. Direct tensor: when batch is already the input tensor

    Args:
        batch_item: The batch item from a data loader
        input_key: Key to use for dictionary format (default: "input_ids")

    Returns:
        The input tensor extracted from the batch
    """
    assert isinstance(batch_item, dict | tuple | Tensor), (
        f"Unsupported batch format: {type(batch_item)}. Must be a dictionary, tuple, or tensor."
    )
    if isinstance(batch_item, dict):
        # Dictionary format: extract the specified key
        if input_key not in batch_item:
            available_keys = list(batch_item.keys())
            raise KeyError(
                f"Key '{input_key}' not found in batch. Available keys: {available_keys}"
            )
        tensor = batch_item[input_key]
    elif isinstance(batch_item, tuple):
        # Assume input is the first element
        tensor = batch_item[0]
    else:
        # Direct tensor format
        tensor = batch_item

    return tensor


def calc_kl_divergence_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    """Calculate the KL divergence between two logits."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl = F.kl_div(log_q, p, reduction="none")  # P · (log P − log Q)
    return kl.sum(dim=-1).mean()  # Σ_vocab / (batch·seq)


def apply_nested_updates(base_dict: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply nested updates to a dictionary."""
    result = copy.deepcopy(base_dict)

    for key, value in updates.items():
        if "." in key:
            # Handle nested keys
            keys = key.split(".")
            current = result

            # Navigate to the parent of the final key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the final value
            current[keys[-1]] = value
        else:
            # Simple key
            result[key] = value

    return result


def runtime_cast[T](type_: type[T], obj: Any) -> T:
    """typecast with a runtime check"""
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, got {type(obj)}")
    return obj
