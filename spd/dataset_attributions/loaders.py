"""Loaders for dataset attributions."""

from pathlib import Path

from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.settings import SPD_OUT_DIR

# Base directory for dataset attributions
DATASET_ATTRIBUTIONS_DIR = SPD_OUT_DIR / "dataset_attributions"


def get_attributions_dir(wandb_run_id: str) -> Path:
    """Get the base dataset attributions directory for a run."""
    return DATASET_ATTRIBUTIONS_DIR / wandb_run_id


def get_attributions_subrun_dir(wandb_run_id: str, subrun_id: str) -> Path:
    """Get the sub-run directory for a specific attribution invocation."""
    return get_attributions_dir(wandb_run_id) / subrun_id


def _find_latest_subrun(wandb_run_id: str) -> Path | None:
    """Find the latest sub-run directory, or None if no sub-runs exist."""
    base_dir = get_attributions_dir(wandb_run_id)
    if not base_dir.exists():
        return None
    candidates = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("da-")],
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def load_dataset_attributions(wandb_run_id: str) -> DatasetAttributionStorage | None:
    """Load dataset attributions from the latest sub-run."""
    subrun = _find_latest_subrun(wandb_run_id)
    if subrun is None:
        return None
    path = subrun / "dataset_attributions.pt"
    if not path.exists():
        return None
    return DatasetAttributionStorage.load(path)
