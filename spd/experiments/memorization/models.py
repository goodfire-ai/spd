"""Memorization model for studying single layer MLPs trained on key-value pairs."""

import json
from pathlib import Path
from typing import Any, ClassVar, Literal, override

import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.log import logger
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class MemorizationPaths(BaseModel):
    """Paths to output files from a Memorization model training run."""

    memorization_train_config: Path
    key_value_pairs: Path
    checkpoint: Path


class MemorizationConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    n_pairs: PositiveInt = Field(description="Number of key-value pairs to memorize")
    d_model: PositiveInt = Field(description="Dimension of keys and values")
    d_hidden: PositiveInt = Field(description="Hidden layer dimension")
    act_fn_name: Literal["gelu", "relu", "none"] = Field(
        description="Activation function. 'none' means no activation (linear layer only)"
    )
    use_bias: bool = Field(default=True, description="Whether to use bias in linear layers")


class SingleLayerMemorizationMLP(nn.Module):
    def __init__(self, config: MemorizationConfig):
        super().__init__()
        self.config = config

        self.linear = nn.Linear(config.d_model, config.d_hidden, bias=config.use_bias)
        self.output = nn.Linear(config.d_hidden, config.d_model, bias=config.use_bias)

        if config.act_fn_name == "gelu":
            self.act_fn = F.gelu
        elif config.act_fn_name == "relu":
            self.act_fn = F.relu
        else:
            self.act_fn = lambda x: x

    @override
    def forward(self, keys: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        hidden = self.linear(keys)
        hidden = self.act_fn(hidden)
        values = self.output(hidden)
        return values

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> MemorizationPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        memorization_train_config_path = download_wandb_file(
            run, run_dir, "memorization_train_config.yaml"
        )
        key_value_pairs_path = download_wandb_file(run, run_dir, "key_value_pairs.json")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        logger.info(f"Downloaded checkpoint from {checkpoint_path}")
        return MemorizationPaths(
            memorization_train_config=memorization_train_config_path,
            key_value_pairs=key_value_pairs_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(
        cls, path: ModelPath
    ) -> tuple[
        "SingleLayerMemorizationMLP",
        dict[str, Any],
        tuple[Float[Tensor, "n_pairs d_model"], Float[Tensor, "n_pairs d_model"]],
    ]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `memorization_train_config.yaml` and `key_value_pairs.json` are in the
                same directory as the checkpoint.

        Returns:
            model: The pretrained SingleLayerMemorizationMLP
            memorization_train_config_dict: The config dict used to train the model
            key_value_pairs: Tuple of (keys, values) tensors used to train the model
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                paths = MemorizationPaths(
                    memorization_train_config=run_dir / "memorization_train_config.yaml",
                    key_value_pairs=run_dir / "key_value_pairs.json",
                    checkpoint=run_dir / "memorization.pth",
                )
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = MemorizationPaths(
                memorization_train_config=Path(path).parent / "memorization_train_config.yaml",
                key_value_pairs=Path(path).parent / "key_value_pairs.json",
                checkpoint=Path(path),
            )

        with open(paths.memorization_train_config) as f:
            memorization_train_config_dict = yaml.safe_load(f)

        with open(paths.key_value_pairs) as f:
            kv_data = json.load(f)
            keys = torch.tensor(kv_data["keys"])
            values = torch.tensor(kv_data["values"])

        # Extract memorization config fields from the flat config structure
        memorization_config_fields = {
            "n_pairs": memorization_train_config_dict.get("n_pairs"),
            "d_model": memorization_train_config_dict.get("d_model"),
            "d_hidden": memorization_train_config_dict.get("d_hidden"),
            "act_fn_name": memorization_train_config_dict.get("act_fn_name", "relu"),
            "use_bias": memorization_train_config_dict.get("use_bias", True),
        }
        memorization_config = MemorizationConfig(**memorization_config_fields)
        model = cls(memorization_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(params)

        return model, memorization_train_config_dict, (keys, values)
