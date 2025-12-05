"""Simple checkpoint save/load for SPD training resumption.

This module provides functions for saving and loading full training checkpoints,
including model state, optimizer state, RNG states, and dataloader position.
Follows the SPD style: simple functions, fail-fast assertions, clear errors.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.run_utils import save_file

# Version for checkpoint format (for future compatibility)
CHECKPOINT_VERSION = "1.0"

# Critical config fields that must match for resume compatibility
CRITICAL_CONFIG_FIELDS = [
    "C",
    "target_module_patterns",
    "pretrained_model_class",
    "ci_fn_type",
    "ci_fn_hidden_dims",
    "sigmoid_type",
    "use_delta_component",
]


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Find the latest checkpoint in a directory by step number.

    Args:
        out_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not out_dir.exists():
        return None

    checkpoints = list(out_dir.glob("model_*.pth"))
    if not checkpoints:
        return None

    def extract_step(path: Path) -> int:
        """Extract step number from filename like 'model_1000.pth'."""
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            logger.warning(f"Could not parse step from checkpoint filename: {path.name}")
            return -1

    latest = max(checkpoints, key=extract_step)
    step = extract_step(latest)

    if step < 0:
        return None

    return latest


def save_checkpoint(
    step: int,
    component_model: ComponentModel,
    optimizer: optim.Optimizer,
    config: Config,
    dataloader_steps_consumed: int,
    out_dir: Path,
) -> Path:
    """Save a full training checkpoint.

    Includes model state, optimizer state (momentum, etc.), RNG states for reproducibility,
    dataloader position, and config snapshot for validation.

    Note: In distributed training, caller is responsible for ensuring this is only called
    from the main process using is_main_process().

    Args:
        step: Current training step
        component_model: The component model to checkpoint
        optimizer: The optimizer to checkpoint
        config: Current training config
        dataloader_steps_consumed: Number of dataloader steps consumed (for skip on resume)
        out_dir: Directory to save checkpoint to

    Returns:
        Path to saved checkpoint file
    """
    # Collect all RNG states
    rng_states = {
        "torch": torch.get_rng_state().cpu(),  # Move to CPU for serialization
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }

    # Add CUDA RNG state if available
    # Store as tuple of tensors (converted from list for type compatibility)
    if torch.cuda.is_available():
        cuda_states = tuple(state.cpu() for state in torch.cuda.get_rng_state_all())
        rng_states["torch_cuda"] = cuda_states

    checkpoint = {
        "step": step,
        "model_state_dict": component_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": rng_states,
        "dataloader_state": {
            "steps_consumed": dataloader_steps_consumed,
        },
        "config_snapshot": config.model_dump(),
        "version": CHECKPOINT_VERSION,
    }

    checkpoint_path = out_dir / f"model_{step}.pth"
    save_file(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    component_model: ComponentModel,
    optimizer: optim.Optimizer,
    config: Config,
) -> tuple[int, int]:
    """Load a checkpoint and restore training state.

    Validates config compatibility (errors on breaking changes, warns on non-critical changes),
    loads model and optimizer state, restores RNG states for reproducibility.

    Args:
        checkpoint_path: Path to checkpoint file
        component_model: Model to load state into
        optimizer: Optimizer to load state into
        config: Current config (for validation)

    Returns:
        Tuple of (checkpoint_step, dataloader_steps_consumed)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint is incompatible with current config
        AssertionError: If checkpoint format is invalid
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint (weights_only=False needed for RNG states and optimizer)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate checkpoint structure
    required_keys = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "rng_states",
        "dataloader_state",
        "config_snapshot",
    }
    missing_keys = required_keys - checkpoint.keys()
    assert not missing_keys, f"Checkpoint missing required keys: {missing_keys}"

    # Validate config compatibility
    saved_config = checkpoint["config_snapshot"]
    _validate_config_compatibility(saved_config, config.model_dump())

    # Load model state
    component_model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model state from step {checkpoint['step']}")

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info("Loaded optimizer state")

    # Restore RNG states for reproducibility
    # Note: States were saved as CPU tensors, so no need to call .cpu() again
    torch.set_rng_state(checkpoint["rng_states"]["torch"])
    if torch.cuda.is_available() and "torch_cuda" in checkpoint["rng_states"]:
        cuda_states = checkpoint["rng_states"]["torch_cuda"]
        # Only set if we have the same number of devices
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            logger.warning(
                f"Saved checkpoint has {len(cuda_states)} CUDA devices, "
                f"but current setup has {torch.cuda.device_count()}. Skipping CUDA RNG restore."
            )
    np.random.set_state(checkpoint["rng_states"]["numpy"])
    random.setstate(checkpoint["rng_states"]["python"])
    logger.info("Restored RNG states")

    # Get dataloader state
    dataloader_steps = checkpoint["dataloader_state"]["steps_consumed"]

    logger.info(f"Successfully loaded checkpoint from step {checkpoint['step']}")

    return checkpoint["step"], dataloader_steps


def _validate_config_compatibility(
    saved_config: dict[str, Any], current_config: dict[str, Any]
) -> None:
    """Validate that current config is compatible with checkpoint's config.

    Errors on breaking changes (architecture differences), warns on non-critical changes
    (hyperparameters that can safely differ).

    Args:
        saved_config: Config from checkpoint
        current_config: Current training config

    Raises:
        ValueError: If configs have incompatible (breaking) differences
    """
    # Check critical fields - these must match exactly
    breaking_changes = []

    for field in CRITICAL_CONFIG_FIELDS:
        if field not in saved_config or field not in current_config:
            # Field missing in one config - skip (could be added field)
            continue

        saved_value = saved_config[field]
        current_value = current_config[field]

        if saved_value != current_value:
            breaking_changes.append(
                f"  {field}:\n    Saved:   {saved_value}\n    Current: {current_value}"
            )

    if breaking_changes:
        changes_str = "\n".join(breaking_changes)
        raise ValueError(
            f"Cannot resume: Config has incompatible architecture changes:\n{changes_str}\n\n"
            f"These fields affect model structure and must match for resume.\n"
            f"If you want to change these, you must start training from scratch."
        )

    # Check non-critical fields - warn but don't block
    non_critical_fields = ["lr", "steps", "batch_size", "seed", "train_log_freq", "eval_freq"]
    non_critical_changes = []

    for field in non_critical_fields:
        if (
            field in saved_config
            and field in current_config
            and saved_config[field] != current_config[field]
        ):
            non_critical_changes.append(
                f"{field}: {saved_config[field]} -> {current_config[field]}"
            )

    if non_critical_changes:
        changes_str = ", ".join(non_critical_changes)
        logger.warning(
            f"Config has non-critical changes from checkpoint: {changes_str}. "
            f"Continuing with current config values."
        )

    logger.info("Config compatibility check passed")
