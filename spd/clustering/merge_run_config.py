"""Configuration for merge clustering runs that combines merge config with run parameters."""

import hashlib
import json
from pathlib import Path
from typing import Self, override

import yaml
from muutils.misc.numerical import shorten_numerical_to_str
from pydantic import Field, PositiveInt, model_validator

from spd.clustering.merge_config import MergeConfig
from spd.clustering.scripts._get_model_path import convert_model_path
from spd.registry import TaskName


class MergeRunConfig(MergeConfig):
    """Configuration for a complete merge clustering run.

    Extends MergeConfig with parameters for model, dataset, and batch configuration.
    CLI-only parameters (base_path, devices, max_concurrency) are intentionally excluded.
    """

    model_path: str = Field(
        description="Path to the model (e.g., wandb run ID or spd_exp: key)",
    )
    task_name: TaskName | None = Field(
        default=None,
        description="Task name for the model. If None, inferred from model_path",
    )
    n_batches: PositiveInt = Field(
        default=10,
        description="Number of batches to split the dataset into (ensemble size)",
    )
    batch_size: PositiveInt = Field(
        default=64,
        description="Size of each batch for processing",
    )

    @model_validator(mode="after")
    def infer_task_name(self) -> Self:
        """Infer task_name from model_path if not provided."""
        if self.task_name is None:
            self.model_path, self.task_name = convert_model_path(self.model_path)

        assert self.task_name in TaskName.__args__, (
            f"Invalid task_name inferred from model_path: {self.task_name = }, must be in {TaskName.__args__ = }"
        )
        return self

    @property
    @override
    def stable_hash(self) -> str:
        """Generate a stable hash including all config parameters."""
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:8]

    @property
    def task_name_validated(self) -> TaskName:
        """Return the validated task name."""
        assert self.task_name is not None, "task_name must be set"
        return self.task_name

    @property
    def run_id(self) -> str:
        """Generate a consistent run ID for this configuration"""
        iters_str: str = shorten_numerical_to_str(self.iters)
        batch_str: str = shorten_numerical_to_str(self.batch_size)
        n_str: str = shorten_numerical_to_str(self.n_batches)

        return f"{self.task_name}-i_{iters_str}-b_{batch_str}-n_{n_str}-h_{self.stable_hash}"

    @classmethod
    def from_file(cls, path: Path) -> "MergeRunConfig":
        """Load config from JSON or YAML file."""
        content: str = path.read_text()

        if path.suffix == ".json":
            return cls.model_validate(json.loads(content))
        elif path.suffix in [".yaml", ".yml"]:
            return cls.model_validate(yaml.safe_load(content))
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

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
