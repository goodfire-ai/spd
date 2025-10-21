"""ClusteringRunConfig"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Self

from pydantic import Field, PositiveInt, model_validator

from spd.base_config import BaseConfig
from spd.clustering.merge_config import MergeConfig
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import SPD_CACHE_DIR


class LoggingIntervals(BaseConfig):
    """Intervals in which to log each type of output."""

    stat: PositiveInt = Field(
        default=1, description="Logging statistics (e.g., k_groups, merge_pair_cost, mdl_loss)"
    )
    tensor: PositiveInt = Field(
        default=100, description="Logging tensors (e.g., wandb_log_tensor, fraction calculations)"
    )
    plot: PositiveInt = Field(
        default=100, description="Generating plots (e.g., plot_merge_iteration)"
    )
    artifact: PositiveInt = Field(
        default=100, description="Creating artifacts (e.g., merge_history)"
    )


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
    base_output_dir: Path = Field(
        default=SPD_CACHE_DIR / "clustering",
        description="Base directory to save clustering runs",
    )
    ensemble_id: str | None = Field(
        default=None,
        description="Ensemble identifier for WandB grouping",
    )
    # TODO: allow idx_in_ensemble to be `None` if ensemble_id is `None`?
    # TODO: allow idx_in_ensemble to be auto-assigned by reading from db if -1?
    idx_in_ensemble: int = Field(0, description="Index of this run in the ensemble")

    merge_config: MergeConfig = Field(description="Merge algorithm configuration")
    logging_intervals: LoggingIntervals = Field(
        default_factory=LoggingIntervals,
        description="Logging intervals",
    )

    wandb_project: str | None = Field(
        default=None,
        description="WandB project name (None to disable WandB logging)",
    )
    wandb_entity: str = Field(default="goodfire", description="WandB entity (team/user) name")
    dataset_streaming: bool = Field(
        default=False,
        description="Whether to use streaming dataset loading (if supported by the dataset). see https://github.com/goodfire-ai/spd/pull/199",
    )

    # TODO: no way to check this without knowing task
    # @model_validator(mode="after")
    # def validate_streaming_compatibility(self) -> Self:
    #     """Ensure dataset_streaming is only enabled for compatible tasks."""
    #     if self.dataset_streaming and self.task_name != "lm":
    #         raise ValueError(
    #             f"Streaming dataset loading only supported for 'lm' task, got '{self.task_name}'"
    #         )
    #     return self

    @model_validator(mode="before")
    def process_experiment_key(cls, values: dict[str, Any]) -> dict[str, Any]:
        experiment_key: str | None = values.get("experiment_key")
        if experiment_key:
            model_path_given: str | None = values.get("model_path")
            model_path_from_experiment: str | None = EXPERIMENT_REGISTRY[
                experiment_key
            ].canonical_run
            assert model_path_from_experiment is not None, (
                f"Experiment '{experiment_key}' has no canonical_run defined in the EXPERIMENT_REGISTRY"
            )
            if model_path_given and model_path_given != model_path_from_experiment:
                raise ValueError(
                    f"Both experiment_key '{experiment_key}' and model_path '{model_path_given}' given in config data, but they disagree: {model_path_from_experiment=}"
                )

            values["model_path"] = model_path_from_experiment
            del values["experiment_key"]

        return values

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

    def stable_hash_b64(self) -> str:
        """Generate a stable, deterministic base64-encoded hash of this config.

        Uses SHA256 hash of the JSON representation with sorted keys for determinism.
        Returns URL-safe base64 encoding without padding.

        Returns:
            URL-safe base64-encoded hash (without padding)
        """
        config_dict: dict[str, Any] = self.model_dump(mode="json")
        config_json: str = json.dumps(config_dict, indent=2, sort_keys=True)
        hash_digest: bytes = hashlib.sha256(config_json.encode()).digest()
        # Use base64 URL-safe encoding and strip padding for filesystem safety
        hash_b64: str = base64.urlsafe_b64encode(hash_digest).decode().rstrip("=")
        return hash_b64
