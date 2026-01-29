from dataclasses import dataclass
from typing import Any, Self, override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.configs import Config
from spd.experiments.tms.configs import TMSModelConfig, TMSTrainConfig
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.spd_types import ModelPath
from spd.utils.module_utils import expand_module_patterns

TMSBatch = tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]
TMSOutput = Float[Tensor, "... n_features"]


@dataclass
class TMSTargetRunInfo(RunInfo[TMSTrainConfig]):
    """Run info from training a TMSModel."""

    config_class = TMSTrainConfig
    config_filename = "tms_train_config.yaml"
    checkpoint_filename = "tms.pth"


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
        self, batch: TMSBatch | Float[Tensor, "... n_features"], **_: Any
    ) -> Float[Tensor, "... n_features"]:
        x = batch[0] if isinstance(batch, tuple) else batch
        hidden = self.linear1(x)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                hidden = layer(hidden)
        out_pre_relu = self.linear2(hidden)
        out = F.relu(out_pre_relu)
        return out

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


def load_tms_component_model_from_run_info(
    run_info: RunInfo[Config],
) -> ComponentModel[TMSBatch, TMSOutput]:
    """Load a trained TMS ComponentModel from a run info object."""
    config = run_info.config
    assert config.pretrained_model_path is not None

    target_model = TMSModel.from_pretrained(config.pretrained_model_path)
    target_model.eval()
    target_model.requires_grad_(False)

    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    comp_model: ComponentModel[TMSBatch, TMSOutput] = ComponentModel(
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


def load_tms_component_model(path: ModelPath) -> ComponentModel[TMSBatch, TMSOutput]:
    """Load a trained TMS ComponentModel from a wandb or local path."""
    run_info = SPDRunInfo.from_path(path)
    return load_tms_component_model_from_run_info(run_info)
