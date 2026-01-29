import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from spd.configs import Config
from spd.experiments.resid_mlp.configs import (
    ResidMLPModelConfig,
    ResidMLPTrainConfig,
)
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.spd_types import ModelPath
from spd.utils.module_utils import expand_module_patterns, init_param_

ResidMLPBatch = tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]
ResidMLPOutput = Float[Tensor, "... n_features"]


@dataclass
class ResidMLPTargetRunInfo(RunInfo[ResidMLPTrainConfig]):
    """Run info from training a ResidualMLPModel."""

    label_coeffs: Float[Tensor, " n_features"]

    config_class = ResidMLPTrainConfig
    config_filename = "resid_mlp_train_config.yaml"
    checkpoint_filename = "resid_mlp.pth"
    extra_files = ["label_coeffs.json"]

    @classmethod
    @override
    def _process_extra_files(cls, file_paths: dict[str, Path], init_kwargs: dict[str, Any]) -> None:
        with open(file_paths["label_coeffs.json"]) as f:
            init_kwargs["label_coeffs"] = torch.tensor(json.load(f))


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


class ResidMLP(LoadableModule):
    def __init__(self, config: ResidMLPModelConfig):
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
        batch: ResidMLPBatch | Float[Tensor, "... n_features"],
        return_residual: bool = False,
    ) -> Float[Tensor, "... n_features"] | Float[Tensor, "... d_embed"]:
        x = batch[0] if isinstance(batch, tuple) else batch
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

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[ResidMLPTrainConfig]) -> "ResidMLP":
        """Load a pretrained model from a run info object."""
        resid_mlp_model = cls(config=run_info.config.resid_mlp_model_config)
        resid_mlp_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return resid_mlp_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ResidMLP":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = ResidMLPTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)


def load_resid_mlp_component_model_from_run_info(
    run_info: RunInfo[Config],
) -> ComponentModel[ResidMLPBatch, ResidMLPOutput]:
    """Load a trained ResidMLP ComponentModel from a run info object."""
    config = run_info.config
    assert config.pretrained_model_path is not None

    target_model = ResidMLP.from_pretrained(config.pretrained_model_path)
    target_model.eval()
    target_model.requires_grad_(False)

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    comp_model: ComponentModel[ResidMLPBatch, ResidMLPOutput] = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        ci_fn_type=config.ci_fn_type,
        sigmoid_type=config.sigmoid_type,
    )

    comp_model_weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
    handle_deprecated_state_dict_keys_(comp_model_weights)
    comp_model.load_state_dict(comp_model_weights)

    return comp_model


def load_resid_mlp_component_model(
    path: ModelPath,
) -> ComponentModel[ResidMLPBatch, ResidMLPOutput]:
    """Load a trained ResidMLP ComponentModel from a wandb or local path."""
    run_info = SPDRunInfo.from_path(path)
    return load_resid_mlp_component_model_from_run_info(run_info)
