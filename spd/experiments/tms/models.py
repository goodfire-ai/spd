from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, override

import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.experiments.tms.configs import TMSModelConfig, TMSTaskConfig, TMSTrainConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.log import logger
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


@dataclass
class TMSTargetRunInfo(RunInfo[TMSTrainConfig]):
    """Run info from training a TMSModel."""

    @override
    @classmethod
    def from_path(cls, path: ModelPath) -> "TMSTargetRunInfo":
        """Load the run info from a wandb run or a local path to a checkpoint."""
        task_name = TMSTaskConfig.model_fields["task_name"].default
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                tms_train_config_path = run_dir / f"{task_name}_train_config.yaml"
                checkpoint_path = run_dir / f"{task_name}.pth"
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                tms_train_config_path, checkpoint_path = TMSModel._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            tms_train_config_path = Path(path).parent / f"{task_name}_train_config.yaml"
            checkpoint_path = Path(path)

        with open(tms_train_config_path) as f:
            tms_train_config_dict = yaml.safe_load(f)

        train_config = TMSTrainConfig(**tms_train_config_dict)
        return cls(checkpoint_path=checkpoint_path, config=train_config)


class TMSModel(LoadableModule):
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

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        self = super().to(*args, **kwargs)
        # Weights will become untied if moving device
        if self.config.tied_weights:
            self.tie_weights_()
        return self

    @override
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
    def _download_wandb_files(wandb_project_run_id: str) -> tuple[Path, Path]:
        """Download the relevant files from a wandb run.

        Returns:
            - tms_train_config_path: Path to the tms_train_config.yaml file
            - checkpoint_path: Path to the checkpoint file
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        task_name = TMSTaskConfig.model_fields["task_name"].default
        tms_train_config_path = download_wandb_file(run, run_dir, f"{task_name}_train_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        logger.info(f"Downloaded checkpoint from {checkpoint_path}")
        return tms_train_config_path, checkpoint_path

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[TMSTrainConfig]) -> "TMSModel":
        """Load a pretrained model from a run info object."""
        tms_model = cls(config=run_info.config.tms_model_config)
        tms_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        tms_model.tie_weights_()
        return tms_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "TMSModel":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = TMSTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
