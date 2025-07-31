"""Utilities for managing experiment run directories and IDs."""

import json
import secrets
import string
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml

from spd.settings import SPD_CACHE_DIR


def get_local_run_id() -> str:
    """Generate a unique run ID. Used if wandb is not active.

    Format: local-<random_8_chars>
    Where random_8_chars is a combination of lowercase letters and digits.

    Returns:
        Unique run ID string
    """
    # Generate 8 random characters (lowercase letters and digits)
    chars = string.ascii_lowercase + string.digits
    random_suffix = "".join(secrets.choice(chars) for _ in range(8))

    return f"local-{random_suffix}"


def get_output_dir(use_wandb_id: bool = True) -> Path:
    """Get the output directory for a run.

    If WandB is active, uses the WandB project and run ID. Otherwise, generates a local run ID.

    Returns:
        Path to the output directory
    """
    # Check if wandb is active and has a run
    if use_wandb_id:
        assert wandb.run is not None, "WandB run is not active"
        # Get project name from wandb.run, fallback to "spd" if not available
        project = getattr(wandb.run, "project", "spd")
        run_id = f"{project}-{wandb.run.id}"
    else:
        run_id = get_local_run_id()

    run_dir = SPD_CACHE_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)


def _save_yaml(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, **kwargs)


def _save_torch(data: Any, path: Path | str, **kwargs: Any) -> None:
    torch.save(data, path, **kwargs)


def _save_text(data: str, path: Path | str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def check_run_exists(wandb_string: str) -> Path | None:
    """Check if a run exists in the shared filesystem based on WandB string.

    Args:
        wandb_string: WandB string in format "wandb:project/runs/run_id"

    Returns:
        Path to the run directory if it exists, None otherwise
    """
    if not wandb_string.startswith("wandb:"):
        return None

    # Parse the wandb string
    parts = wandb_string.replace("wandb:", "").split("/")
    if len(parts) != 3 or parts[1] != "runs":
        return None

    project = parts[0]
    run_id = parts[2]

    # Check if directory exists with format project-runid
    run_dir = SPD_CACHE_DIR / "runs" / f"{project}-{run_id}"
    return run_dir if run_dir.exists() else None


def save_file(data: dict[str, Any] | Any, path: Path | str, **kwargs: Any) -> None:
    """Save a file.

    NOTE: This function was originally designed to save files with specific permissions,
    bypassing the system's umask. This is not needed anymore, but we're keeping this
    abstraction for convenience and brevity.

    File type is determined by extension:
    - .json: Save as JSON
    - .yaml/.yml: Save as YAML
    - .pth/.pt: Save as PyTorch model
    - .txt or other: Save as plain text (data must be string)

    Args:
        data: Data to save (format depends on file type)
        path: File path to save to
        **kwargs: Additional arguments passed to the specific save function
    """
    path = Path(path)
    suffix = path.suffix.lower()

    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        _save_json(data, path, **kwargs)
    elif suffix in [".yaml", ".yml"]:
        _save_yaml(data, path, **kwargs)
    elif suffix in [".pth", ".pt"]:
        _save_torch(data, path, **kwargs)
    else:
        # Default to text file
        assert isinstance(data, str), f"For {suffix} files, data must be a string, got {type(data)}"
        _save_text(data, path, encoding=kwargs.get("encoding", "utf-8"))
