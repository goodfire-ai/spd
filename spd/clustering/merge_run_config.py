"""Configuration for merge clustering runs that combines merge config with run parameters."""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from muutils.misc.numerical import shorten_numerical_to_str
from pydantic import BaseModel, Field, PositiveInt, model_validator

from spd.clustering.merge_config import MergeConfig
from spd.registry import EXPERIMENT_REGISTRY, ExperimentConfig
from spd.spd_types import TaskName

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


class RunFilePaths:
    def __init__(self, run_path: Path):
        self.run_path = run_path

    @property
    def histories_path(self) -> Path:
        return self.run_path / "merge_histories"

    @property
    def distances_dir(self) -> Path:
        return self.run_path / "distances"

    @property
    def run_config_path(self) -> Path:
        return self.run_path / "run_config.json"

    def scaffold(self) -> None:
        self.histories_path.mkdir(exist_ok=True)
        self.distances_dir.mkdir(exist_ok=True)


class RunConfig(BaseModel):
    """Configuration for a complete merge clustering run.

    Extends MergeConfig with parameters for model, dataset, and batch configuration.
    CLI-only parameters (base_path, devices, max_concurrency) are intentionally excluded.
    """

    merge_config: MergeConfig = Field(
        description="Merge configuration",
    )

    model_path: str = Field(
        description="WandB path to the model (format: wandb:entity/project/run_id)",
    )
    task_name: TaskName = Field(
        description="Task name for the model (must be explicit)",
    )
    experiment_key: str | None = Field(
        default=None,
        description="Original experiment key if created from spd_exp registry",
    )
    n_batches: PositiveInt = Field(
        default=10,
        description="Number of batches to split the dataset into (ensemble size)",
    )
    batch_size: PositiveInt = Field(
        default=64,
        description="Size of each batch for processing",
    )

    # Implementation details:
    base_path: Path = Field(
        ...,
        description="Base path for saving clustering outputs",
    )
    workers_per_device: int = Field(
        ...,
        description="Maximum number of concurrent clustering processes per device",
    )
    devices: list[str] = Field(
        ...,
        description="Devices to use for clustering",
    )

    # WandB configuration
    wandb_enabled: bool = Field(
        default=False,
        description="Enable WandB logging for clustering runs",
    )
    wandb_project: str = Field(
        default="spd-cluster",
        description="WandB project name for clustering runs",
    )
    intervals: dict[IntervalKey, PositiveInt] = Field(
        default_factory=lambda: _DEFAULT_INTERVALS.copy(),
        description="Intervals for different logging operations",
    )

    @model_validator(mode="after")
    def validate_model_path(self) -> Self:
        """Validate that model_path is a proper WandB path."""
        if not self.model_path.startswith("wandb:"):
            raise ValueError(f"model_path must start with 'wandb:', got: {self.model_path}")

        assert self.task_name in TaskName.__args__, (
            f"Invalid task_name: {self.task_name = }, must be in {TaskName.__args__ = }"
        )
        return self

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        """Ensure all required interval keys are present."""
        # warning if any keys are missing
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
        """Extract the WandB run ID of the source decomposition from the model_path

        Format: wandb:entity/project/run_id or wandb:entity/project/runs/run_id
        """
        parts: list[str] = self.model_path.replace("wandb:", "").split("/")
        if len(parts) >= 3:
            # Handle both formats: with and without 'runs' in path
            return parts[-1] if parts[-1] != "runs" else parts[-2] if len(parts) > 3 else parts[-1]
        else:
            raise ValueError(f"Invalid wandb path format: {self.model_path}")

    @property
    def wandb_group(self) -> str:
        """Generate WandB group name based on parent model"""
        return f"model-{self.wandb_decomp_model}"

    @property
    def _iters_str(self) -> str:
        """Shortened string representation of iterations for run ID"""
        return shorten_numerical_to_str(self.merge_config.iters)

    @property
    def config_identifier(self) -> str:
        """Unique identifier for this specific config on this specific model.

        Format: model_abc123-a0.1-i1k-b64-n10-h_12ab
        Allows filtering in WandB for all runs with this exact config and model.
        """
        return f"task_{self.task_name}-w_{self.wandb_decomp_model}-a{self.merge_config.alpha:g}-i{self._iters_str}-b{self.batch_size}-n{self.n_batches}-h_{self.stable_hash}"

    @property
    def stable_hash(self) -> str:
        """Generate a stable hash including all config parameters."""
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]

    @classmethod
    def from_file(cls, path: Path) -> "RunConfig":
        """Load config from JSON or YAML file.

        Handles legacy spd_exp: model_path format and enforces consistency.
        """
        # read the file contents, load them according to extension
        content: str = path.read_text()
        data: dict[str, Any]
        if path.suffix == ".json":
            data = json.loads(content)
        elif path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        # if we provide an experiment_key, then:
        # 1. use the `EXPERIMENT_REGISTRY` to fill in model_path and task_name
        # 2. check it's consistent with model_path and task_name from the file if those are provided
        experiment_key: str | None = data.get("experiment_key")
        model_path: str | None = data.get("model_path")
        task_name: str | None = data.get("task_name")
        if experiment_key is not None:
            exp_config: ExperimentConfig = EXPERIMENT_REGISTRY[experiment_key]

            # Enforce consistency if explicit fields present
            if model_path is not None:
                assert model_path == exp_config.canonical_run, (
                    f"Inconsistent model_path for {experiment_key}, version from file ({model_path}) does not match registry ({exp_config.canonical_run})"
                )
            if task_name is not None:
                assert task_name == exp_config.task_name, (
                    f"Inconsistent task_name for {experiment_key}, version from file ({task_name}) does not match registry ({exp_config.task_name})"
                )

            # overwrite in data dict
            data["model_path"] = exp_config.canonical_run
            data["task_name"] = exp_config.task_name

        return cls.model_validate(data)
