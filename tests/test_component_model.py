import tempfile
from pathlib import Path
from typing import Any, override

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D as RadfordConv1D

from spd.configs import Config
from spd.experiments.tms.configs import TMSTaskConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.mask_info import make_mask_infos
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import EmbeddingComponents, LinearComponents
from spd.spd_types import ModelPath
from spd.utils.run_utils import save_file


class SimpleTestModel(LoadableModule):
    """Simple test model with Linear and Embedding layers for unit‑testing."""

    LINEAR_1_SHAPE = (10, 5)
    LINEAR_2_SHAPE = (1, 3)
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
        x = self.linear1(x)
        x = self.conv1d1(x)
        x = self.conv1d2(x)
        x = self.linear2(x)
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


def test_components_mode_with_empty_masks_matches_target(component_model: ComponentModel):
    cm = component_model
    x = torch.randn(2, 10)
    out_target = cm(x, mode="target")
    out_components = cm(x, mode="components", mask_infos=make_mask_infos({}))
    torch.testing.assert_close(out_components, out_target, rtol=1e-5, atol=1e-6)


def _two_linear_model() -> nn.Module:
    class TwoLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(4, 4, bias=False)
            self.linear2 = nn.Linear(4, 2, bias=False)

        @override
        def forward(self, x: Float[Tensor, "... 4"]) -> Float[Tensor, "... 2"]:
            return self.linear2(self.linear1(x))

    m = TwoLinear()
    m.requires_grad_(False)
    return m


def test_components_mode_with_delta_matches_target() -> None:
    # Use a simple 2-layer model to avoid conv/embedding complexity
    tiny = _two_linear_model()
    cm = ComponentModel(
        target_model=tiny,
        target_module_patterns=["linear1", "linear2"],
        C=3,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )
    B = 5
    x = torch.randn(B, 4)
    out_target = cm(x, mode="target")

    # Build masks and deltas that reconstruct the target outputs
    masks = {name: torch.ones(B, cm.C) for name in cm.components}
    deltas = {
        name: (cm.get_original_weight(name) - cm.components[name].weight, torch.ones(B))
        for name in cm.components
    }
    out_components = cm(x, mode="components", mask_infos=make_mask_infos(masks, deltas))
    torch.testing.assert_close(out_components, out_target, rtol=1e-5, atol=1e-6)


def test_pre_forward_cache_collects_inputs() -> None:
    tiny = _two_linear_model()
    cm = ComponentModel(
        target_model=tiny,
        target_module_patterns=["linear1", "linear2"],
        C=3,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )
    x = torch.randn(2, 4)
    out, cache = cm.forward(x, mode="pre_forward_cache", module_names=["linear1", "linear2"])
    assert isinstance(out, torch.Tensor)
    assert set(cache.keys()) == {"linear1", "linear2"}
    assert cache["linear1"].shape == torch.Size([2, 4])
    assert cache["linear2"].shape == torch.Size([2, 4])


def test_linear_components_forward_matches_mask_math():
    B = 5
    C = 3
    input_dim = 6
    output_dim = 4

    components = LinearComponents(d_in=input_dim, d_out=output_dim, C=C, bias=None)
    x = torch.randn(B, input_dim)
    mask = torch.rand(B, C)
    out_rep = components(x, mask)
    # Sanity: changing mask to zeros yields zeros (since no bias)
    out_zero = components(x, torch.zeros_like(mask))
    assert torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-6)
    assert not torch.allclose(out_rep, out_zero)


def test_linear_components_broadcast_over_sequence():
    B = 5
    S = 7
    C = 3
    input_dim = 6
    output_dim = 4
    components = LinearComponents(d_in=input_dim, d_out=output_dim, C=C, bias=None)
    x = torch.randn(B, S, input_dim)
    mask = torch.rand(B, S, C)
    y = components(x, mask)
    assert y.shape == torch.Size([B, S, output_dim])


def test_embedding_components_forward_matches_mask_math():
    vocab_size = 50
    embedding_dim = 16
    C = 2
    comp = EmbeddingComponents(vocab_size=vocab_size, embedding_dim=embedding_dim, C=C)
    batch_size = 4
    seq_len = 7
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.rand(batch_size, seq_len, C)
    out_rep = comp(idx, mask)
    out_zero = comp(idx, torch.zeros_like(mask))
    assert torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-6)
    assert not torch.allclose(out_rep, out_zero)


def _make_identity_linear_components(d: int) -> LinearComponents:
    comp = LinearComponents(C=d, d_in=d, d_out=d, bias=None)
    with torch.no_grad():
        comp.V.copy_(torch.eye(d))
        comp.U.copy_(torch.eye(d))
    return comp


def test_identity_linear_component_behaves_like_identity() -> None:
    B = 5
    d = 10
    identity_comp = _make_identity_linear_components(d)
    x = torch.randn(B, d)
    y = identity_comp(x, mask=torch.ones(B, d))
    torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-5)


def test_identity_linear_component_broadcast_over_sequence() -> None:
    B = 3
    S = 7
    d = 6
    identity_comp = _make_identity_linear_components(d)
    x = torch.randn(B, S, d)
    y = identity_comp(x, mask=torch.ones(B, S, d))
    torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-5)


def test_components_and_identity_sum_is_correct() -> None:
    # Identity + components with mask ones should equal linear comp output plus x
    B = 4
    d_in = 6
    d_out = 3
    components = LinearComponents(C=3, d_in=d_in, d_out=d_out, bias=None)
    identity_comp = _make_identity_linear_components(d_in)
    x = torch.randn(B, d_in)
    mask = torch.rand(B, 3)
    y_comp = components(x, mask)
    y_id = identity_comp(x, mask=torch.ones(B, d_in))
    assert y_id.shape == x.shape
    assert y_comp.shape == torch.Size([B, d_out])


def test_linear_components_shapes() -> None:
    B = 2
    S = 5
    d_in = 4
    d_out = 3
    C = 2
    components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
    x = torch.randn(B, S, d_in)
    mask = torch.rand(B, S, C)
    y = components(x, mask)
    assert y.shape == torch.Size([B, S, d_out])


def test_correct_parameters_require_grad(component_model: ComponentModel):
    cm = component_model
    # target model frozen
    for p in cm.target_model.parameters():
        assert not p.requires_grad
    # components trainable (except bias buffer)
    for comp in cm.components.values():
        if isinstance(comp, LinearComponents | EmbeddingComponents):
            assert comp.U.requires_grad
            assert comp.V.requires_grad


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
