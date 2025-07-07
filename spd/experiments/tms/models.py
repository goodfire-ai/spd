from __future__ import annotations

from pathlib import Path
from typing import Any
from typing_extensions import Self

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class TMSModelPaths(BaseModel):
    """Paths to output files from a TMSModel training run."""

    tms_train_config: Path
    checkpoint: Path


class TMSModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    tied_weights: bool
    init_bias_to_zero: bool
    device: str


class TMSModel(nn.Module):
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config

        self.linear1 = nn.Linear(config.n_features, config.n_hidden, bias=False)
        self.linear2 = nn.Linear(config.n_hidden, config.n_features, bias=True)
        if config.init_bias_to_zero:
            self.linear2.bias.data.zero_()

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList()
            for _ in range(config.n_hidden_layers):
                layer = nn.Linear(config.n_hidden, config.n_hidden, bias=False)
                self.hidden_layers.append(layer)

        if config.tied_weights:
            self.tie_weights_()

    def tie_weights_(self) -> None:
        self.linear2.weight.data = self.linear1.weight.data.T

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self = super().to(*args, **kwargs)
        # Weights will become untied if moving device
        if self.config.tied_weights:
            self.tie_weights_()
        return self

    def forward(
        self, x: Float[Tensor, "... n_features"], **_: Any
    ) -> Float[Tensor, "... n_features"]:
        hidden = self.linear1(x)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                hidden = layer(hidden)
        out_pre_relu = self.linear2(hidden)
        out = F.relu(out_pre_relu)
        return out

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TMSModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        tms_model_config_path = download_wandb_file(run, run_dir, "tms_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TMSModelPaths(tms_train_config=tms_model_config_path, checkpoint=checkpoint_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TMSModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained TMSModel
            tms_model_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = TMSModelPaths(
                tms_train_config=Path(path).parent / "tms_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.tms_train_config) as f:
            tms_train_config_dict = yaml.safe_load(f)

        tms_config = TMSModelConfig(**tms_train_config_dict["tms_model_config"])
        tms = cls(config=tms_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        tms.load_state_dict(params)

        if tms_config.tied_weights:
            tms.tie_weights_()

        return tms, tms_train_config_dict
