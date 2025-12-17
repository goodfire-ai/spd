"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.harvest.schemas import DATA_ROOT

# Base directory for autointerp data (separate from harvest)
AUTOINTERP_DATA_DIR = DATA_ROOT / "autointerp"


def get_autointerp_dir(wandb_run_id: str) -> Path:
    """Get the autointerp (interpretations) directory for a run."""
    return AUTOINTERP_DATA_DIR / wandb_run_id


@dataclass
class ArchitectureInfo:
    n_blocks: int
    c: int
    model_class: str
    dataset_name: str
    tokenizer_name: str


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
