"""CLI script for computing max-activating text samples for language model component clusters."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from spd.settings import REPO_ROOT


# TODO: BaseModel -> BaseConfig once #200 is merged
class DashboardConfig(BaseModel):
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

    @classmethod
    def read(cls, config_path: Path) -> "DashboardConfig":
        """Load dashboard config from JSON or YAML file.

        Args:
            config_path: Path to config file (.json or .yaml)

        Returns:
            Loaded DashboardConfig
        """
        import yaml

        if config_path.suffix == ".json":
            config_dict = json.loads(config_path.read_text())
        elif config_path.suffix in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls.model_validate(config_dict)
