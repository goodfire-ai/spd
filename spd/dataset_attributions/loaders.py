"""Loaders for dataset attributions."""

from pathlib import Path

from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.settings import SPD_OUT_DIR

# Base directory for dataset attributions
DATASET_ATTRIBUTIONS_DIR = SPD_OUT_DIR / "dataset_attributions"


def get_attributions_dir(wandb_run_id: str) -> Path:
    """Get the dataset attributions directory for a run."""
    return DATASET_ATTRIBUTIONS_DIR / wandb_run_id


def load_dataset_attributions(wandb_run_id: str) -> DatasetAttributionStorage | None:
    """Load dataset attributions, if available."""
    path = get_attributions_dir(wandb_run_id) / "dataset_attributions.pt"
    if not path.exists():
        return None
    return DatasetAttributionStorage.load(path)
