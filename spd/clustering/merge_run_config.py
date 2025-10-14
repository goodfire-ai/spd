"""ClusteringRunConfig"""

from pathlib import Path
from typing import Any, Self

from pydantic import Field, PositiveInt, model_validator

from spd.base_config import BaseConfig
from spd.clustering.merge_config import MergeConfig
from spd.settings import SPD_CACHE_DIR
from spd.utils.run_utils import get_local_run_id


class LoggingIntervals(BaseConfig):
    """Intervals in which to log each type of output."""

    stat: PositiveInt = Field(
        1, description="Logging statistics (e.g., k_groups, merge_pair_cost, mdl_loss)"
    )
    tensor: PositiveInt = Field(
        100, description="Logging tensors (e.g., wandb_log_tensor, fraction calculations)"
    )
    plot: PositiveInt = Field(100, description="Generating plots (e.g., plot_merge_iteration)")
    artifact: PositiveInt = Field(100, description="Creating artifacts (e.g., merge_history)")


class ClusteringRunConfig(BaseConfig):
    """Configuration for a single clustering run.

    This config specifies the clustering algorithm parameters and data processing settings.
    Deployment concerns (where to save, WandB settings, ensemble configuration) are handled
    by ClusteringSubmitConfig.
    """

    # TODO: Handle both wandb strings and local file paths
    model_path: str = Field(
        description="WandB path to the decomposed model (format: wandb:entity/project/run_id)"
    )

    batch_size: PositiveInt = Field(..., description="Batch size for processing")
    dataset_seed: int = Field(0, description="Seed for dataset generation/loading")
    idx_in_ensemble: int = Field(0, description="Index of this run in the ensemble")
    base_output_dir: Path = Field(
        default=SPD_CACHE_DIR / "clustering",
        description="Base directory to save clustering runs",
    )
    ensemble_id: str = Field(
        default_factory=get_local_run_id,
        description="Ensemble identifier for WandB grouping",
    )

    merge_config: MergeConfig = Field(description="Merge algorithm configuration")

    wandb_project: str | None = Field(
        default=None,
        description="WandB project name (None to disable WandB logging)",
    )
    wandb_entity: str = Field(default="goodfire", description="WandB entity (team/user) name")

    logging_intervals: LoggingIntervals = Field(..., description="Logging intervals")

    @model_validator(mode="after")
    def validate_model_path(self) -> Self:
        """Validate that model_path is a proper WandB path."""
        if not self.model_path.startswith("wandb:"):
            raise ValueError(f"model_path must start with 'wandb:', got: {self.model_path}")
        return self

    @property
    def wandb_decomp_model(self) -> str:
        """Extract the WandB run ID of the source decomposition."""
        parts = self.model_path.replace("wandb:", "").split("/")
        if len(parts) >= 3:
            return parts[-1] if parts[-1] != "runs" else parts[-2]
        raise ValueError(f"Invalid wandb path format: {self.model_path}")

    def model_dump_with_properties(self) -> dict[str, Any]:
        """Serialize config including computed properties for WandB logging."""
        base_dump: dict[str, Any] = self.model_dump(mode="json")

        # Add computed properties
        base_dump.update(
            {
                "wandb_decomp_model": self.wandb_decomp_model,
            }
        )

        return base_dump
