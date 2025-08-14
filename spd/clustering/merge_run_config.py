"""Configuration for merge clustering runs that combines merge config with run parameters."""

import hashlib
import json
from pathlib import Path
from typing import Any, Self, override

import yaml
from muutils.misc.numerical import shorten_numerical_to_str
from pydantic import Field, PositiveInt, model_validator

from spd.clustering.merge_config import MergeConfig
from spd.registry import EXPERIMENT_REGISTRY, ExperimentConfig, TaskName


class MergeRunConfig(MergeConfig):
    """Configuration for a complete merge clustering run.

    Extends MergeConfig with parameters for model, dataset, and batch configuration.
    CLI-only parameters (base_path, devices, max_concurrency) are intentionally excluded.
    """

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

    # WandB configuration
    wandb_enabled: bool = Field(
        default=False,
        description="Enable WandB logging for clustering runs",
    )
    wandb_project: str = Field(
        default="spd-cluster",
        description="WandB project name for clustering runs",
    )
    wandb_log_frequency: PositiveInt = Field(
        default=1,
        description="Log metrics to WandB every N iterations",
    )
    wandb_artifact_frequency: PositiveInt = Field(
        default=100,
        description="Save GroupMerge artifacts to WandB every N iterations",
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

    @property
    def wandb_decomp_model(self) -> str:
        """Extract the WandB run ID of the source decomposition from the model_path"""
        # Format: wandb:entity/project/run_id or wandb:entity/project/runs/run_id
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
        return shorten_numerical_to_str(self.iters)

    @property
    def config_identifier(self) -> str:
        """Unique identifier for this specific config on this specific model.

        Format: model_abc123-a0.1-i1k-b64-n10-h_12ab
        Allows filtering in WandB for all runs with this exact config and model.
        """
        return f"task_{self.task_name}-w_{self.wandb_decomp_model}-a{self.alpha:g}-i{self._iters_str}-b{self.batch_size}-n{self.n_batches}-h_{self.stable_hash}"

    @property
    @override
    def stable_hash(self) -> str:
        """Generate a stable hash including all config parameters."""
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]

    @classmethod
    def from_file(cls, path: Path) -> "MergeRunConfig":
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

    def to_file(self, path: Path) -> None:
        """Save config to file (format inferred from extension)."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            path.write_text(self.model_dump_json(indent=2))
        elif path.suffix in [".yaml", ".yml"]:
            path.write_text(
                yaml.dump(
                    self.model_dump(mode="json"),
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    def to_merge_config(self) -> MergeConfig:
        """Extract the base MergeConfig from this instance."""
        return MergeConfig(**{field: getattr(self, field) for field in MergeConfig.model_fields})

    def model_dump_with_properties(self) -> dict[str, Any]:
        """Serialize config including computed properties for WandB logging."""
        base_dump: dict[str, Any] = self.model_dump()
        
        # Add computed properties
        base_dump.update({
            "wandb_decomp_model": self.wandb_decomp_model,
            "wandb_group": self.wandb_group,
            "config_identifier": self.config_identifier,
            "stable_hash": self.stable_hash,
        })
        
        return base_dump
