"""Configuration for component dashboard generation."""

from pathlib import Path

from pydantic import Field

from spd.base_config import BaseConfig
from spd.settings import REPO_ROOT


class ComponentDashboardConfig(BaseConfig):
    """Configuration for minimal component dashboard generation."""

    # Model and output
    model_path: str = Field(description="Path to SPD model (wandb: or local path)")
    output_dir: Path = Field(
        default=REPO_ROOT / "spd/dashboard/data",
        description="Output directory for dashboard data",
    )

    # Dataset configuration
    dataset_name: str = Field(
        default="SimpleStories/SimpleStories",
        description="HuggingFace dataset name",
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use",
    )
    dataset_column: str = Field(
        default="story",
        description="Column name containing text",
    )
    dataset_streaming: bool = Field(
        default=False,
        description="Whether to use streaming dataset loading",
    )

    # Data generation parameters
    batch_size: int = Field(
        default=64,
        description="Batch size for data loading",
    )
    n_batches: int = Field(
        default=10,
        description="Number of batches to process",
    )
    context_length: int = Field(
        default=64,
        description="Context length for tokenization",
    )

    # Component filtering
    dead_threshold: float = Field(
        default=1e-6,
        description="Components with max activation <= this are considered dead",
    )

    # Analysis parameters
    n_samples: int = Field(
        default=16,
        description="Number of top samples to collect per component",
    )
    embed_dim: int = Field(
        default=2,
        description="Dimensionality of component embeddings",
    )
    hist_bins: int = Field(
        default=50,
        description="Number of bins for histograms",
    )

    # Token statistics parameters
    token_stats_top_n: int = Field(
        default=5,
        description="Number of top tokens to store per component",
    )
    token_active_threshold: float = Field(
        default=0.01,
        description="Threshold above which a component is considered active",
    )
