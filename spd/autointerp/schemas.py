"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.settings import SPD_OUT_DIR

# Base directory for autointerp data
AUTOINTERP_DATA_DIR = SPD_OUT_DIR / "autointerp"


def get_autointerp_dir(wandb_run_id: str) -> Path:
    """Get the top-level autointerp directory for an SPD run."""
    return AUTOINTERP_DATA_DIR / wandb_run_id


def get_autointerp_run_dir(wandb_run_id: str, autointerp_run_id: str) -> Path:
    """Get the directory for a specific autointerp run (timestamped subdirectory)."""
    return AUTOINTERP_DATA_DIR / wandb_run_id / autointerp_run_id


@dataclass
class ArchitectureInfo:
    n_blocks: int
    c_per_layer: dict[str, int]
    model_class: str
    dataset_name: str
    tokenizer_name: str
    layer_descriptions: dict[str, str]


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
    prompt: str
