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
    ComponentMaskInfo,
    ComponentsOrModule,
    EmbeddingComponents,
    LinearComponents,
    make_mask_infos,
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


def test_no_replacement_masks_means_original_mode(component_model: ComponentModel):
    cm = component_model

    # Initial state: nothing should be active
    for comp in cm.components_or_modules.values():
        comp.assert_pristine()
    # No masks supplied: everything should stay in "original" mode
    with cm._replaced_modules({}):
        assert all(comp.forward_mode == "original" for comp in cm.components_or_modules.values())
        assert all(comp.components_mask is None for comp in cm.components_or_modules.values())
    # After the context the state must be fully reset
    for comp in cm.components_or_modules.values():
        comp.assert_pristine()


def test_replaced_modules_sets_and_restores_masks(component_model: ComponentModel):
    cm = component_model
    full_mask_infos = {
        name: ComponentMaskInfo(
            routing_mask=True,
            component_mask=torch.randn(1, cm.C, dtype=torch.float32),
            weight_delta_and_mask=None,
        )
        for name in cm.components_or_modules
    }
    with cm._replaced_modules(full_mask_infos):
        # All components should now be in replacement‑mode with the given masks
        for name, comp in cm.components_or_modules.items():
            assert comp.forward_mode == "components"
            assert torch.equal(comp.components_mask, full_mask_infos[name].component_mask)  # pyright: ignore [reportArgumentType]

    for comp in cm.components_or_modules.values():
        comp.assert_pristine()


def test_replaced_modules_sets_and_restores_masks_partial(
    component_model: ComponentModel,
):
    cm = component_model
    # Partial masking
    partial_masks = {
        "linear1": ComponentMaskInfo(
            routing_mask=True,
            component_mask=torch.ones(1, cm.C),
            weight_delta_and_mask=None,
        )
    }
    with cm._replaced_modules(partial_masks):
        assert isinstance((fw := cm.components_or_modules["linear1"].forward_mode), tuple)
        mode, mask_info = fw
        assert mode == "mixed"
        assert torch.equal(mask_info.component_mask, partial_masks["linear1"].component_mask)  # pyright: ignore[reportArgumentType]

        # Others fall back to target‑only mode with no masks
        assert cm.components_or_modules["linear2"].forward_mode == "target"
        assert cm.components_or_modules["embedding"].forward_mode == "target"

    for comp in cm.components_or_modules.values():
        comp.assert_pristine()


def test_replaced_modules_sets_and_restores_identity_masks_only(
    component_model_with_identity: ComponentModel,
) -> None:
    cm = component_model_with_identity
    identity_masks = {"identity_linear1": torch.ones(1, cm.C)}
    with cm._replaced_modules(make_mask_infos(identity_masks)):
        for name, comp in cm.components_or_modules.items():
            if name == "linear1":
                assert comp.forward_mode == "components"
                assert comp.components_mask is None
                assert torch.equal(comp.identity_mask, identity_masks["identity_linear1"])  # pyright: ignore [reportArgumentType]
            else:
                assert comp.forward_mode == "original"
                assert comp.components_mask is None
                assert comp.identity_mask is None

    for comp in cm.components_or_modules.values():
        comp.assert_pristine()


def test_replaced_modules_sets_and_restores_combined_masks(
    component_model_with_identity: ComponentModel,
) -> None:
    cm = component_model_with_identity
    masks = {
        "linear1": torch.randn(1, cm.C),
        "identity_linear1": torch.ones(1, cm.C),
        "identity_conv1d1": torch.ones(1, cm.C),
    }

    with cm._replaced_modules(make_mask_infos(masks)):
        for name, comp in cm.components_or_modules.items():
            if name == "linear1":
                assert comp.forward_mode == "components"
                assert torch.equal(comp.components_mask, masks["linear1"])  # pyright: ignore [reportArgumentType]
                assert torch.equal(comp.identity_mask, masks["identity_linear1"])  # pyright: ignore [reportArgumentType]
            elif name == "conv1d1":
                assert comp.forward_mode == "components"
                assert comp.components_mask is None
                assert torch.equal(comp.identity_mask, masks["identity_conv1d1"])  # pyright: ignore [reportArgumentType]
            else:
                assert comp.forward_mode == "original"
                assert comp.components_mask is None
                assert comp.identity_mask is None

    for comp in cm.components_or_modules.values():
        comp.assert_pristine()


def test_replaced_component_forward_linear_matches_modes():
    B = 5
    C = 3
    input_dim = 6
    output_dim = 4

    target = nn.Linear(input_dim, output_dim, bias=True)
    components = LinearComponents(d_in=input_dim, d_out=output_dim, C=3, bias=target.bias)
    components_or_module = ComponentsOrModule(target=target, components=components)

    x = torch.randn(B, input_dim)

    # --- target path ---
    components_or_module.forward_mode = "target"
    out_orig = components_or_module(x)
    expected_orig = target(x)
    torch.testing.assert_close(out_orig, expected_orig, rtol=1e-4, atol=1e-5)

    # --- Replacement path (with mask) ---
    mask = torch.rand(B, C)
    mask_info = ComponentMaskInfo(
        routing_mask=True, component_mask=mask, weight_delta_and_mask=None
    )
    components_or_module.forward_mode = ("mixed", mask_info)
    out_rep = components_or_module(x)
    expected_rep = components(x, mask)
    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)


def test_replaced_component_forward_conv1d_matches_modes():
    B = 5
    S = 10
    C = 3
    input_dim = 6
    output_dim = 4

    original = RadfordConv1D(nf=output_dim, nx=input_dim)

    components = LinearComponents(d_in=input_dim, d_out=output_dim, C=C, bias=original.bias)
    components_or_module = ComponentsOrModule(target=original, components=components)

    x = torch.randn(B, S, input_dim)

    # --- Original path ---
    components_or_module.forward_mode = "target"
    out_orig = components_or_module(x)
    expected_orig = original(x)

    torch.testing.assert_close(out_orig, expected_orig, rtol=1e-4, atol=1e-5)

    # --- Replacement path (with mask) ---
    mask = torch.rand(B, S, C)  # (B, L, C)
    mask_info = ComponentMaskInfo(
        routing_mask=True, component_mask=mask, weight_delta_and_mask=None
    )
    components_or_module.forward_mode = ("mixed", mask_info)
    out_rep = components_or_module(x)
    expected_rep = components(x, mask)

    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)


def test_replaced_component_forward_embedding_matches_modes():
    vocab_size = 50
    embedding_dim = 16
    C = 2

    emb = nn.Embedding(vocab_size, embedding_dim)
    comp = EmbeddingComponents(vocab_size=vocab_size, embedding_dim=embedding_dim, C=C)
    rep = ComponentsOrModule(target=emb, components=comp)

    batch_size = 4
    seq_len = 7
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))  # (batch pos)

    # --- Target path ---
    rep.forward_mode = "target"
    out_orig = rep(idx)
    expected_orig = emb(idx)
    torch.testing.assert_close(out_orig, expected_orig, rtol=1e-4, atol=1e-5)

    # --- Replacement path (with mask) ---
    mask = torch.rand(batch_size, seq_len, C)  # (batch pos C)
    mask_info = ComponentMaskInfo(
        routing_mask=True, component_mask=mask, weight_delta_and_mask=None
    )
    rep.forward_mode = ("mixed", mask_info)
    out_rep = rep(idx)
    expected_rep = comp.forward(idx, mask)
    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)


def test_correct_parameters_require_grad(component_model: ComponentModel):
    for cm in component_model.components_or_modules.values():
        if isinstance(cm.target, nn.Linear | RadfordConv1D):
            assert not cm.target.weight.requires_grad
            if cm.target.bias is not None:  # pyright: ignore [reportUnnecessaryComparison]
                assert not cm.target.bias.requires_grad
            assert isinstance(cm.components, LinearComponents)
            if cm.components.bias is not None:
                assert not cm.components.bias.requires_grad
            assert cm.components.U.requires_grad
            assert cm.components.V.requires_grad
        else:
            assert isinstance(cm.target, nn.Embedding), "sanity check"
            assert not cm.target.weight.requires_grad
            assert isinstance(cm.components, EmbeddingComponents)
            assert cm.components.U.requires_grad
            assert cm.components.V.requires_grad


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


def test_identity_module_patterns_embedding_raises() -> None:
    raise NotImplementedError("attach to identity injection logic, might not even go here")


# Don't THINK this is needed anymore
#
# def test_patch_modules_components_only() -> None:
#     model = SimpleTestModel()
#     model.requires_grad_(False)

#     C = 4
#     module_paths = ["linear1", "embedding", "conv1d1"]
#     identity_module_paths: list[str] = []

#     patched, replaced = ComponentModel._patch_modules(
#         model=model, module_paths=module_paths, identity_module_paths=identity_module_paths, C=C
#     )

#     # Only the specified modules should be replaced
#     assert isinstance(patched.get_submodule("linear1"), ComponentsOrModule)
#     assert isinstance(patched.get_submodule("embedding"), ComponentsOrModule)
#     assert isinstance(patched.get_submodule("conv1d1"), ComponentsOrModule)
#     assert isinstance(patched.linear2, nn.Linear)
#     assert isinstance(patched.conv1d2, RadfordConv1D)
#     assert isinstance(patched.other_layer, nn.ReLU)

#     # Keys in returned dict should match all replaced paths
#     assert set(replaced.keys()) == set(module_paths)

#     # linear1: components present, identity_components absent; dims and bias preserved
#     rep_lin1 = patched.get_submodule("linear1")
#     assert isinstance(rep_lin1, ComponentsOrModule)
#     assert rep_lin1.components is not None and rep_lin1.identity_components is None
#     assert isinstance(rep_lin1.components, LinearComponents)
#     assert isinstance(rep_lin1.target, nn.Linear)
#     d_out_lin1, d_in_lin1 = rep_lin1.target.weight.shape  # (d_out, d_in)
#     assert rep_lin1.components.d_in == d_in_lin1
#     assert rep_lin1.components.d_out == d_out_lin1
#     comp_bias = rep_lin1.components.bias
#     assert comp_bias is not None
#     assert torch.equal(comp_bias, rep_lin1.target.bias.data)

#     # embedding: EmbeddingComponents with correct dims
#     rep_emb = patched.get_submodule("embedding")
#     assert isinstance(rep_emb, ComponentsOrModule)
#     assert rep_emb.components is not None and rep_emb.identity_components is None
#     assert isinstance(rep_emb.components, EmbeddingComponents)
#     assert isinstance(rep_emb.target, nn.Embedding)
#     assert rep_emb.components.vocab_size == rep_emb.target.num_embeddings
#     assert rep_emb.components.embedding_dim == rep_emb.target.embedding_dim

#     # conv1d1: LinearComponents with conv dims interpreted correctly
#     rep_conv = patched.get_submodule("conv1d1")
#     assert isinstance(rep_conv, ComponentsOrModule)
#     assert rep_conv.components is not None and rep_conv.identity_components is None
#     assert isinstance(rep_conv.components, LinearComponents)
#     assert isinstance(rep_conv.target, RadfordConv1D)
#     d_in_conv, d_out_conv = rep_conv.target.weight.shape  # Conv1D stores (nx, nf)
#     assert rep_conv.components.d_in == d_in_conv
#     assert rep_conv.components.d_out == d_out_conv


def test_patch_modules_unsupported_component_type_raises() -> None:
    model = SimpleTestModel()
    model.requires_grad_(False)

    with pytest.raises(ValueError):
        ComponentModel._patch_modules(
            model=model,
            module_paths=["other_layer"],
            C=2,
        )
