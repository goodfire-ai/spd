import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import einops
import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.experiments.resid_mlp.configs import ResidualMLPModelConfig, ResidualMLPTrainConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.log import logger
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.module_utils import init_param_
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class ResidualMLPPaths(BaseModel):
    """Paths to output files from a ResidualMLPModel training run."""

    resid_mlp_train_config: Path
    label_coeffs: Path
    checkpoint: Path


@dataclass
class ResidualMLPTargetRunInfo(RunInfo[ResidualMLPTrainConfig]):
    """Run info from training a ResidualMLPModel."""

    label_coeffs: Float[Tensor, " n_features"]

    @override
    @classmethod
    def from_path(cls, path: ModelPath) -> "ResidualMLPTargetRunInfo":
        """Load the run info from a wandb run or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                paths = ResidualMLPPaths(
                    resid_mlp_train_config=run_dir / "resid_mlp_train_config.yaml",
                    label_coeffs=run_dir / "label_coeffs.json",
                    checkpoint=run_dir / "resid_mlp.pth",
                )
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                paths = ResidualMLP._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = ResidualMLPPaths(
                resid_mlp_train_config=Path(path).parent / "resid_mlp_train_config.yaml",
                label_coeffs=Path(path).parent / "label_coeffs.json",
                checkpoint=Path(path),
            )

        with open(paths.resid_mlp_train_config) as f:
            resid_mlp_train_config_dict = yaml.safe_load(f)

        with open(paths.label_coeffs) as f:
            label_coeffs = torch.tensor(json.load(f))

        resid_mlp_train_config = ResidualMLPTrainConfig(**resid_mlp_train_config_dict)
        return cls(
            checkpoint_path=paths.checkpoint,
            config=resid_mlp_train_config,
            label_coeffs=label_coeffs,
        )


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.act_fn = act_fn

        self.mlp_in = nn.Linear(d_model, d_mlp, bias=in_bias)
        self.mlp_out = nn.Linear(d_mlp, d_model, bias=out_bias)

    @override
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        mid_pre_act_fn = self.mlp_in(x)
        mid = self.act_fn(mid_pre_act_fn)
        out = self.mlp_out(mid)
        return out


class ResidualMLP(LoadableModule):
    def __init__(self, config: ResidualMLPModelConfig):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty(config.n_features, config.d_embed))
        init_param_(self.W_E, fan_val=config.n_features, nonlinearity="linear")
        self.W_U = nn.Parameter(torch.empty(config.d_embed, config.n_features))
        init_param_(self.W_U, fan_val=config.d_embed, nonlinearity="linear")

        assert config.act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu
        self.layers = nn.ModuleList(
            [
                MLP(
                    d_model=config.d_embed,
                    d_mlp=config.d_mlp,
                    act_fn=self.act_fn,
                    in_bias=config.in_bias,
                    out_bias=config.out_bias,
                )
                for _ in range(config.n_layers)
            ]
        )

    @override
    def forward(
        self,
        x: Float[Tensor, "... n_features"],
        return_residual: bool = False,
    ) -> Float[Tensor, "... n_features"] | Float[Tensor, "... d_embed"]:
        residual = einops.einsum(x, self.W_E, "... n_features, n_features d_embed -> ... d_embed")
        for layer in self.layers:
            out = layer(residual)
            residual = residual + out
        if return_residual:
            return residual
        out = einops.einsum(
            residual,
            self.W_U,
            "... d_embed, d_embed n_features -> ... n_features",
        )
        return out

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> ResidualMLPPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        resid_mlp_train_config_path = download_wandb_file(
            run, run_dir, "resid_mlp_train_config.yaml"
        )
        label_coeffs_path = download_wandb_file(run, run_dir, "label_coeffs.json")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        logger.info(f"Downloaded checkpoint from {checkpoint_path}")
        return ResidualMLPPaths(
            resid_mlp_train_config=resid_mlp_train_config_path,
            label_coeffs=label_coeffs_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[ResidualMLPTrainConfig]) -> "ResidualMLP":
        """Load a pretrained model from a run info object."""
        resid_mlp_model = cls(config=run_info.config.resid_mlp_config)
        resid_mlp_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return resid_mlp_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ResidualMLP":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = ResidualMLPTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
