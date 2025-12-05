import importlib
import random
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from pydantic import BaseModel, PositiveFloat
from pydantic.v1.utils import deep_update
from torch import Tensor

from spd.base_config import BaseConfig
from spd.log import logger
from spd.utils.run_utils import save_file

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


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def generate_sweep_id() -> str:
    """Generate a unique sweep ID based on timestamp."""
    return f"sweep_id-{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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
        return (
            lambda step, steps: 1.0
            if steps == 1
            else 0.1 + (1 - 0.1) * 0.5*(1+np.cos(np.pi * step / (steps - 1)))
            # else np.cos(0.5*np.pi * step / (steps - 1))

        )
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
            "spd.experiments.resid_mlp.models.ResidualMLP"
    """
    module_path, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def extract_batch_data(
    batch_item: dict[str, Any] | tuple[Tensor, "..."] | Tensor,
    input_key: str = "input_ids",
) -> Tensor:
    """Extract input data from various batch formats.

    This utility function handles different batch formats commonly used across the codebase:
    1. Dictionary format: {"input_ids": Tensor, "..."} - common in LM tasks
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
    reduce: bool = True,
) -> Float[Tensor, ""] | Float[Tensor, "..."]:
    """Calculate the KL divergence between two logits.

    Args:
        pred: The predicted logits
        target: The target logits
        reduce: Whether to reduce the KL divergence across the batch and sequence dimensions

    Returns:
        The KL divergence
    """
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl_raw = F.kl_div(log_q, p, reduction="none")  # P · (log P − log Q)
    kl = kl_raw.sum(dim=-1)
    if reduce:
        return kl.mean()  # Σ_vocab / (batch·seq)
    else:
        return kl


def calc_sum_recon_loss_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
    loss_type: Literal["mse", "kl", "mem"],
) -> Float[Tensor, ""]:
    """Calculate the reconstruction loss for a language model without reduction.

    Args:
        pred: The predicted logits
        target: The target logits
        loss_type: The type of loss to compute:
            - "mse": Mean squared error across all positions
            - "kl": KL divergence across all positions
            - "mem": KL divergence at the final sequence position only (for mem task)

    Returns:
        The summed loss (not yet divided by number of examples)
    """
    match loss_type:
        case "mse":
            loss = ((pred - target) ** 2).sum()
        case "kl":
            loss = calc_kl_divergence_lm(pred=pred, target=target, reduce=False).sum()
        case "mem":
            # Only compute KL at the final sequence position, restricted to top target vocab item
            # pred/target shape: [batch, seq_len, vocab] -> take [:, -1, :]
            pred_final = pred[:, -1, :]  # [batch, vocab]
            target_final = target[:, -1, :]  # [batch, vocab]
            log_pred = torch.log_softmax(pred_final, dim=-1)
            log_target = torch.log_softmax(target_final, dim=-1)
            loss=((log_pred - log_target)**2).sum()
            # loss = calc_kl_divergence_lm(pred=pred_final, target=target_final, reduce=False).sum()
    return loss


def runtime_cast[T](type_: type[T], obj: Any) -> T:
    """typecast with a runtime check"""
    if not isinstance(obj, type_):
        raise TypeError(f"Expected {type_}, got {type(obj)}")
    return obj


def fetch_latest_checkpoint_name(filenames: list[str], prefix: str | None = None) -> str:
    """Fetch the latest checkpoint name from a list of .pth files.

    Assumes format is <name>_<step>.pth or <name>.pth.
    """
    if prefix:
        filenames = [filename for filename in filenames if filename.startswith(prefix)]
    if not filenames:
        raise ValueError(f"No files found with prefix {prefix}")
    if len(filenames) == 1:
        latest_checkpoint_name = filenames[0]
    else:
        latest_checkpoint_name = sorted(
            filenames, key=lambda x: int(x.split(".pth")[0].split("_")[-1])
        )[-1]
    return latest_checkpoint_name


def fetch_latest_local_checkpoint(run_dir: Path, prefix: str | None = None) -> Path:
    """Fetch the latest checkpoint from a local run directory."""
    filenames = [file.name for file in run_dir.iterdir() if file.name.endswith(".pth")]
    latest_checkpoint_name = fetch_latest_checkpoint_name(filenames, prefix)
    latest_checkpoint_local = run_dir / latest_checkpoint_name
    return latest_checkpoint_local


def save_pre_run_info(
    save_to_wandb: bool,
    out_dir: Path,
    spd_config: BaseConfig,
    sweep_params: dict[str, Any] | None,
    target_model: nn.Module | None,
    train_config: BaseConfig | None,
    task_name: str | None,
) -> None:
    """Save run information locally and optionally to wandb."""

    files_to_save = {
        "final_config.yaml": spd_config.model_dump(mode="json"),
    }

    if target_model is not None:
        files_to_save[f"{task_name}.pth"] = target_model.state_dict()

    if train_config is not None:
        files_to_save[f"{task_name}_train_config.yaml"] = train_config.model_dump(mode="json")

    if sweep_params is not None:
        files_to_save["sweep_params.yaml"] = sweep_params

    for filename, data in files_to_save.items():
        filepath = out_dir / filename
        save_file(data, filepath)

        if save_to_wandb:
            wandb.save(str(filepath), base_path=out_dir, policy="now")


class _HasDevice(Protocol):
    """Protocol for objects with a `.device` attribute that is a `torch.device`."""

    device: torch.device


CanGetDevice = (
    nn.Module
    | _HasDevice
    | Tensor
    | dict[str, Tensor]
    | dict[str, _HasDevice]
    | Sequence[Tensor]
    | Sequence[_HasDevice]
)


def _get_obj_devices(d: CanGetDevice) -> set[torch.device]:
    """try to get the set of devices on which an object's parameters are located"""
    if hasattr(d, "device"):
        # pyright doesn't realize that we just checked for a `.device` attribute, hence the ignores
        assert isinstance(d.device, torch.device)  # pyright: ignore[reportAttributeAccessIssue]
        return {d.device}  # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(d, nn.Module):
        return {param.device for param in d.parameters()}
    elif isinstance(d, dict):
        return {obj.device for obj in d.values()}
    else:
        # this might fail, we don't really know what `d` is at this point
        return {obj.device for obj in d}  # pyright: ignore[reportGeneralTypeIssues]


def get_obj_device(d: CanGetDevice) -> torch.device:
    """Try to get the device of an object's parameters. Asserts that all parameters are on the same device."""
    devices: set[torch.device] = _get_obj_devices(d)
    assert len(devices) == 1, f"Object parameters are on multiple devices: {devices}"
    return devices.pop()


def dict_safe_update_(d1: dict[str, Any], d2: dict[str, Any]) -> None:
    """Update a dictionary with another dictionary, but only if the key is not already present in
    the first dictionary."""
    assert not set(d1.keys()) & set(d2.keys()), "The dictionaries must have no overlapping keys"
    d1.update(d2)
