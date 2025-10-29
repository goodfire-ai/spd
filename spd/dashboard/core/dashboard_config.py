"""CLI script for computing max-activating text samples for language model component clusters."""

from pathlib import Path

from pydantic import Field

from spd.base_config import BaseConfig
from spd.settings import REPO_ROOT


class DashboardConfig(BaseConfig):
    wandb_run: str = Field(description="WandB clustering run path (e.g., entity/project/run_id)")
    output_dir: Path = Field(
        default=REPO_ROOT / "spd/clustering/dashboard/data",
        description="Base output directory (default: {{REPO_ROOT}}/spd/clustering/dashboard/data/)",
    )
    iteration: int = Field(
        default=-1,
        description="Merge iteration to analyze (negative indexes from end, default: -1 for latest)",
    )
    n_samples: int = Field(
        default=16,
        description="Number of top-activating samples to collect per cluster",
    )
    n_batches: int = Field(
        default=4,
        description="Number of data batches to process",
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for data loading",
    )
    context_length: int = Field(
        default=64,
        description="Context length for tokenization",
    )
    write_html: bool = Field(
        default=False,
        description="Write bundled HTML files to output directory",
    )
    dataset_streaming: bool = Field(
        default=False,
        description="Whether to use streaming dataset loading. recommended True for large datasets or tests.",
    )
