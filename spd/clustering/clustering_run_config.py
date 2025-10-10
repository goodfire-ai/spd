"""Configuration for a single clustering run (not an ensemble)."""

import uuid
import warnings
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import Field, PositiveInt, model_validator

from spd.clustering.merge_config import MergeConfig
from spd.settings import SPD_CACHE_DIR
from spd.spd_types import TaskName
from spd.utils.general_utils import BaseConfig

# Define interval types and defaults
IntervalKey = Literal["stat", "tensor", "plot", "artifact"]

IntervalsDict = dict[IntervalKey, PositiveInt]
"""Type alias for intervals dictionary

- `stat`: logging statistics (e.g., k_groups, merge_pair_cost, mdl_loss)
- `tensor`: logging tensors (e.g., wandb_log_tensor, fraction calculations)
- `plot`: generating plots
- `artifact`: creating artifacts (checkpoints)

"""

_DEFAULT_INTERVALS: IntervalsDict = {
    "stat": 1,
    "tensor": 100,
    "plot": 100,
    "artifact": 100,
}


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

    batch_size: PositiveInt = Field(default=64, description="Batch size for processing")
    dataset_seed: int = Field(0, description="Seed for dataset generation/loading")
    idx_in_ensemble: int = Field(0, description="Index of this run in the ensemble")
    output_dir: Path = Field(
        SPD_CACHE_DIR / "clustering" / "clustering_runs",
        description="Directory to save merge history",
    )
    ensemble_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Ensemble identifier for WandB grouping",
    )

    merge_config: MergeConfig = Field(description="Merge algorithm configuration")

    wandb_project: str | None = Field(
        default="spd-cluster",
        description="WandB project name (None to disable WandB logging)",
    )
    wandb_entity: str = Field(default="goodfire", description="WandB entity (team/user) name")

    intervals: IntervalsDict = Field(
        default_factory=lambda: _DEFAULT_INTERVALS.copy(), description="Logging intervals"
    )

    @model_validator(mode="after")
    def validate_model_path(self) -> Self:
        """Validate that model_path is a proper WandB path."""
        if not self.model_path.startswith("wandb:"):
            raise ValueError(f"model_path must start with 'wandb:', got: {self.model_path}")
        return self

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        """Ensure all required interval keys are present."""
        missing_keys: set[IntervalKey] = set(_DEFAULT_INTERVALS.keys()) - set(self.intervals.keys())
        if missing_keys:
            warnings.warn(
                f"Missing interval keys in {self.intervals = }: {missing_keys}. Using defaults for those.",
                UserWarning,
                stacklevel=1,
            )

        self.intervals = {
            **_DEFAULT_INTERVALS,
            **self.intervals,
        }

        return self

    @property
    def wandb_decomp_model(self) -> str:
        """Extract the WandB run ID of the source decomposition."""
        parts = self.model_path.replace("wandb:", "").split("/")
        if len(parts) >= 3:
            return parts[-1] if parts[-1] != "runs" else parts[-2]
        raise ValueError(f"Invalid wandb path format: {self.model_path}")

    def get_task_name(self) -> TaskName:
        """Extract task_name from the SPD model's config.

        This loads the SPD run config and returns the task_name from it.
        Maps "induction_head" to "ih" for compatibility with TaskName type.
        """
        from spd.models.component_model import SPDRunInfo

        spd_run = SPDRunInfo.from_path(self.model_path)
        task_name_raw = spd_run.config.task_config.task_name

        # Map full name to short name
        task_name_map = {
            "induction_head": "ih",
            "tms": "tms",
            "resid_mlp": "resid_mlp",
            "lm": "lm",
        }

        task_name = task_name_map.get(task_name_raw, task_name_raw)

        assert task_name in TaskName.__args__, (
            f"Invalid task_name: {task_name = }, must be in {TaskName.__args__ = }"
        )
        return cast(TaskName, task_name)

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
