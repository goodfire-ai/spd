import tempfile
from pathlib import Path
from typing import Any, override

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers.modeling_utils import Conv1D as RadfordConv1D

from spd.configs import Config
from spd.experiments.tms.configs import TMSTaskConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import (
    ComponentsMaskInfo,
    EmbeddingComponents,
    LinearComponents,
)
from spd.spd_types import ModelPath
from spd.utils.run_utils import save_file


class SimpleTestModel(LoadableModule):
    """Simple test model with Linear and Embedding layers for unit‑testing."""

    LINEAR_1_SHAPE = (10, 5)
    LINEAR_2_SHAPE = (5, 3)
    CONV1D_1_SHAPE = (3, 5)
    CONV1D_2_SHAPE = (1, 3)
    EMBEDDING_SHAPE = (100, 8)

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(*self.LINEAR_1_SHAPE, bias=True)
        self.linear2 = nn.Linear(*self.LINEAR_2_SHAPE, bias=False)
        self.conv1d1 = RadfordConv1D(*self.CONV1D_1_SHAPE)
        self.conv1d2 = RadfordConv1D(*self.CONV1D_2_SHAPE)

        self.embedding = nn.Embedding(*self.EMBEDDING_SHAPE)
        self.other_layer = nn.ReLU()  # Non‑target layer (should never be wrapped)

    @override
    def forward(self, x: Float[Tensor, "... 10"]):  # noqa: D401,E501
        x = self.linear2(self.linear1(x))
        x = self.conv1d2(self.conv1d1(x))
        return x

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[Any]) -> "SimpleTestModel":
        model = cls()
        model.load_state_dict(torch.load(run_info.checkpoint_path))
        return model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "SimpleTestModel":
        model = cls()
        model.load_state_dict(torch.load(path))
        return model


@pytest.fixture(scope="function")
def component_model() -> ComponentModel:
    """Return a fresh ``ComponentModel`` for each test."""
    target_model = SimpleTestModel()
    target_model.requires_grad_(False)
    return ComponentModel(
        target_model=target_model,
        target_module_patterns=["linear1", "linear2", "embedding", "conv1d1", "conv1d2"],
        C=4,
        gate_type="mlp",
        gate_hidden_dims=[4],
        pretrained_model_output_attr=None,
    )

def test_correct_parameters_require_grad(component_model: ComponentModel):
    for module_path, components in component_model.components.items():
        target_module = component_model.target_model.get_submodule(module_path)

        if isinstance(target_module, nn.Linear | RadfordConv1D):
            assert not target_module.weight.requires_grad
            if target_module.bias is not None:  # pyright: ignore [reportUnnecessaryComparison]
                assert not target_module.bias.requires_grad
            assert isinstance(components, LinearComponents)
            if components.bias is not None:
                assert not components.bias.requires_grad
            assert components.U.requires_grad
            assert components.V.requires_grad
        else:
            assert isinstance(target_module, nn.Embedding), "sanity check"
            assert not target_module.weight.requires_grad
            assert isinstance(components, EmbeddingComponents)
            assert components.U.requires_grad
            assert components.V.requires_grad


def test_from_run_info():
    target_model = SimpleTestModel()
    target_model.eval()
    target_model.requires_grad_(False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        base_model_dir = base_dir / "test_model"
        base_model_dir.mkdir(parents=True, exist_ok=True)
        comp_model_dir = base_dir / "comp_model"
        comp_model_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = base_model_dir / "model.pth"
        save_file(target_model.state_dict(), base_model_path)

        config = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            target_module_patterns=["linear1", "linear2", "embedding", "conv1d1", "conv1d2"],
            C=4,
            gate_type="mlp",
            gate_hidden_dims=[4],
            batch_size=1,
            steps=1,
            lr=1e-3,
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            importance_minimality_coeff=1.0,
            pnorm=1.0,
            n_examples_until_dead=1,
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        cm = ComponentModel(
            target_model=target_model,
            target_module_patterns=["linear1", "linear2", "embedding", "conv1d1", "conv1d2"],
            C=4,
            gate_type="mlp",
            gate_hidden_dims=[4],
            pretrained_model_output_attr=None,
        )

        save_file(cm.state_dict(), comp_model_dir / "model.pth")
        save_file(config.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        cm_run_info = SPDRunInfo.from_path(comp_model_dir / "model.pth")
        cm_loaded = ComponentModel.from_run_info(cm_run_info)

        assert config == cm_run_info.config
        for k, v in cm_loaded.state_dict().items():
            torch.testing.assert_close(v, cm.state_dict()[k])


def test_patch_modules_unsupported_component_type_raises() -> None:
    model = SimpleTestModel()
    model.requires_grad_(False)

    with pytest.raises(ValueError):
        ComponentModel._create_components(
            model=model,
            module_paths=["other_layer"],
            C=2,
        )
