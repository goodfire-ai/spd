import tempfile
from pathlib import Path
from typing import Any, override

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers.modeling_utils import Conv1D as RadfordConv1D

from spd.configs import Config, ImportanceMinimalityLossTrainConfig
from spd.experiments.tms.configs import TMSTaskConfig
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.models.components import (
    ComponentsMaskInfo,
    EmbeddingComponents,
    LinearComponents,
    MLPGates,
    ParallelLinear,
    VectorMLPGates,
    VectorSharedMLPGate,
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


def test_correct_parameters_require_grad():
    target_model = SimpleTestModel()
    target_model.requires_grad_(False)

    component_model = ComponentModel(
        target_model=target_model,
        target_module_patterns=["linear1", "linear2", "embedding", "conv1d1", "conv1d2"],
        C=4,
        gate_type="mlp",
        gate_hidden_dims=[4],
        pretrained_model_output_attr=None,
    )

    for module_path, components in component_model.components.items():
        assert components.U.requires_grad
        assert components.V.requires_grad

        target_module = component_model.target_model.get_submodule(module_path)

        if isinstance(target_module, nn.Linear | RadfordConv1D):
            assert not target_module.weight.requires_grad
            if target_module.bias is not None:  # pyright: ignore [reportUnnecessaryComparison]
                assert not target_module.bias.requires_grad
            assert isinstance(components, LinearComponents)
            if components.bias is not None:
                assert not components.bias.requires_grad
        else:
            assert isinstance(target_module, nn.Embedding), "sanity check"
            assert isinstance(components, EmbeddingComponents)
            assert not target_module.weight.requires_grad


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
            identity_module_patterns=["linear1"],
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
            loss_metric_configs=[ImportanceMinimalityLossTrainConfig(coeff=1.0, pnorm=1.0)],
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

        if config.identity_module_patterns is not None:
            insert_identity_operations_(
                target_model, identity_patterns=config.identity_module_patterns
            )

        cm = ComponentModel(
            target_model=target_model,
            target_module_patterns=config.all_module_patterns,
            C=config.C,
            gate_type=config.gate_type,
            gate_hidden_dims=config.gate_hidden_dims,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )

        save_file(cm.state_dict(), comp_model_dir / "model.pth")
        save_file(config.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        cm_run_info = SPDRunInfo.from_path(comp_model_dir / "model.pth")
        cm_loaded = ComponentModel.from_run_info(cm_run_info)

        assert config == cm_run_info.config
        for k, v in cm_loaded.state_dict().items():
            torch.testing.assert_close(v, cm.state_dict()[k])


class TinyTarget(nn.Module):
    def __init__(
        self,
        vocab_size: int = 7,
        d_emb: int = 5,
        d_mid: int = 4,
        d_out: int = 3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_emb)
        self.mlp = nn.Linear(d_emb, d_mid)
        self.out = nn.Linear(d_mid, d_out)

    @override
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "..."]:
        x = self.embed(token_ids)
        x = self.mlp(x)
        x = self.out(x)
        return x


def tiny_target():
    tt = TinyTarget()
    tt.eval()
    tt.requires_grad_(False)
    return tt


BATCH_SIZE = 2


def test_patch_modules_unsupported_component_type_raises() -> None:
    model = tiny_target()
    wrong_module_path = "other_layer"

    with pytest.raises(AttributeError):
        ComponentModel._create_components(
            target_model=model,
            module_paths=[wrong_module_path],
            C=2,
        )


def test_parallel_linear_shapes_and_forward():
    C = 3
    d_in = 4
    d_out = 5
    layer = ParallelLinear(C, d_in, d_out, nonlinearity="relu")
    x = torch.randn(BATCH_SIZE, C, d_in)
    y = layer(x)
    assert y.shape == (BATCH_SIZE, C, d_out)


@pytest.mark.parametrize("hidden_dims", [[8], [4, 3]])
def test_mlp_gates_scalar_per_component(hidden_dims: list[int]):
    C = 5
    gates = MLPGates(C=C, hidden_dims=hidden_dims)
    x = torch.randn(BATCH_SIZE, C)  # two items, C components
    y = gates(x)
    assert y.shape == (BATCH_SIZE, C)


@pytest.mark.parametrize("hidden_dims", [[4], [6, 3]])
def test_vector_mlp_gates(hidden_dims: list[int]):
    C = 3
    d_in = 10
    gates = VectorMLPGates(C=C, input_dim=d_in, hidden_dims=hidden_dims)
    x = torch.randn(BATCH_SIZE, d_in)
    y = gates(x)
    assert y.shape == (BATCH_SIZE, C)


@pytest.mark.parametrize("hidden_dims", [[], [7], [8, 5]])
def test_vector_shared_mlp_gate(hidden_dims: list[int]):
    C = 3
    d_in = 10
    gate = VectorSharedMLPGate(C=C, input_dim=d_in, hidden_dims=hidden_dims)
    x = torch.randn(BATCH_SIZE, d_in)
    y = gate(x)
    assert y.shape == (BATCH_SIZE, C)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # components
        ["target_model.a.b.components.U", "_components.a-b.U"],
        ["target_model.a.b.components.V", "_components.a-b.V"],
        ["target_model.a.b.components.bias", "_components.a-b.bias"],
        # components (old naming)
        ["patched_model.a.b.components.U", "_components.a-b.U"],
        ["patched_model.a.b.components.V", "_components.a-b.V"],
        ["patched_model.a.b.components.bias", "_components.a-b.bias"],
        # original
        ["target_model.a.b.original.weight", "target_model.a.b.weight"],
        # regular state
        ["target_model.a.b.c.weight", "target_model.a.b.c.weight"],
    ],
)
def test_handle_deprecated_state_dict_keys_(key: str, expected: str):
    input_dict = {key: torch.tensor([])}
    handle_deprecated_state_dict_keys_(input_dict)
    expected_dict = {expected: torch.tensor([])}
    assert input_dict.keys() == expected_dict.keys()


def test_full_weight_delta_matches_target_behaviour():
    # GIVEN a component model
    target_model = tiny_target()

    target_module_paths = ["embed", "mlp", "out"]
    cm = ComponentModel(
        target_model=target_model,
        target_module_patterns=target_module_paths,
        C=4,
        gate_type="mlp",
        gate_hidden_dims=[4],
        pretrained_model_output_attr=None,
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    # WHEN we forward the component model with weight deltas and a weight delta mask of all 1s
    weight_deltas = cm.calc_weight_deltas()
    component_masks = {name: torch.ones(BATCH_SIZE, cm.C) for name in target_module_paths}
    weight_deltas_and_masks = {
        name: (weight_deltas[name], torch.ones(BATCH_SIZE)) for name in target_module_paths
    }
    mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=weight_deltas_and_masks)
    out = cm(token_ids, mask_infos=mask_infos)

    # THEN the output matches the target model's output
    torch.testing.assert_close(out, target_model(token_ids))


def test_input_cache_captures_pre_weight_input():
    target_model = tiny_target()

    # GIVEN a component model
    target_module_paths = ["embed", "mlp"]

    cm = ComponentModel(
        target_model=target_model,
        target_module_patterns=target_module_paths,
        C=2,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )

    # WHEN we forward the component model with input caching
    token_ids = torch.randint(
        low=0,
        high=target_model.embed.num_embeddings,
        size=(BATCH_SIZE,),
        dtype=torch.long,
    )
    out, cache = cm(token_ids, cache_type="input")

    # Output isn't altered
    torch.testing.assert_close(out, target_model(token_ids))

    # Captured inputs match the true pre-weight inputs

    assert cache["embed"].dtype == torch.long
    assert torch.equal(cache["embed"], token_ids)
    embed_out = target_model.embed(token_ids)

    assert cache["mlp"].shape == (BATCH_SIZE, target_model.mlp.in_features)
    torch.testing.assert_close(cache["mlp"], embed_out)


def test_weight_deltas():
    # GIVEN a component model
    target_model = tiny_target()
    target_module_paths = ["embed", "mlp", "out"]
    cm = ComponentModel(
        target_model=target_model,
        target_module_patterns=target_module_paths,
        C=3,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )

    # THEN the weight deltas match the target weight
    deltas = cm.calc_weight_deltas()
    for name in target_module_paths:
        target_w = cm.target_weight(name)
        comp_w = cm.components[name].weight
        torch.testing.assert_close(target_w, comp_w + deltas[name])


def test_replacement_effects_fwd_pass():
    d_in = 10
    d_out = 20
    C = 30

    class OneLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias=False)

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    model = OneLayerModel()
    model.eval()
    model.requires_grad_(False)

    cm = ComponentModel(
        target_model=model,
        target_module_patterns=["linear"],
        C=C,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )

    # WHEN we set the target model weights to be UV
    model.linear.weight.copy_(cm.components["linear"].weight)

    # AND we use all components
    input = torch.randn(BATCH_SIZE, d_in)
    use_all_components = ComponentsMaskInfo(component_mask=torch.ones(BATCH_SIZE, C))

    # THEN the model output matches the component model output
    model_out = model(input)
    cm_out_with_all_components = cm(input, mask_infos={"linear": use_all_components})
    torch.testing.assert_close(model_out, cm_out_with_all_components)

    # however, WHEN we double the values of the model weights
    model.linear.weight.mul_(2)

    # THEN the component-only output should be 1/2 the model output
    new_model_out = model(input)
    new_cm_out_with_all_components = cm(input, mask_infos={"linear": use_all_components})
    torch.testing.assert_close(new_model_out, new_cm_out_with_all_components * 2)


def test_replacing_identity():
    d = 10
    C = 20

    class IdentityLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.linear.weight)

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # GIVEN a simple model that performs identity (so we can isolate the effects below)
    model = IdentityLayerModel()
    model.eval()
    model.requires_grad_(False)

    # with another prepended identity layer
    insert_identity_operations_(target_model=model, identity_patterns=["linear"])

    # wrapped in a component model that decomposes the prepended identity layer
    cm = ComponentModel(
        target_model=model,
        target_module_patterns=["linear.pre_identity"],
        C=C,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )

    # and a random input
    input = torch.randn(BATCH_SIZE, d)

    # WHEN we forward with the model
    # THEN it should just act as the identity
    torch.testing.assert_close(model(input), input)
    torch.testing.assert_close(cm(input), input)

    # WHEN we forward with the identity components
    use_all_components = ComponentsMaskInfo(component_mask=torch.ones(BATCH_SIZE, C))

    cm_components_out = cm(input, mask_infos={"linear.pre_identity": use_all_components})

    # THEN it should modify the input
    assert not torch.allclose(cm_components_out, input)

    # BUT the original model output should be unchanged
    cm_target_out = cm(input)
    assert torch.allclose(cm_target_out, model(input))


def test_routing():
    d = 10
    C = 20

    class IdentityLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.linear.weight)

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # GIVEN a simple model that performs identity (so we can isolate the effects below)
    model = IdentityLayerModel()
    model.eval()
    model.requires_grad_(False)

    # wrapped in a component model that decomposes the layer
    cm = ComponentModel(
        target_model=model,
        target_module_patterns=["linear"],
        C=C,
        gate_type="mlp",
        gate_hidden_dims=[2],
        pretrained_model_output_attr=None,
    )

    # and a random input
    input = torch.randn(BATCH_SIZE, d)

    # WHEN we forward with the model
    # THEN it should just act as the identity
    torch.testing.assert_close(model(input), input)
    torch.testing.assert_close(cm(input), input)

    # WHEN we forward with the components
    use_all_components = ComponentsMaskInfo(component_mask=torch.ones(BATCH_SIZE, C))

    cm_components_out = cm(input, mask_infos={"linear": use_all_components})

    # THEN it should modify the input
    assert not torch.allclose(cm_components_out, input)

    # but WHEN we forward with the components with routing:
    use_all_components_for_example_0 = ComponentsMaskInfo(
        component_mask=torch.ones(BATCH_SIZE, C),
        routing_mask=torch.tensor([True, False]),  # route to components only for example 0
    )

    cm_routed_out = cm(input, mask_infos={"linear": use_all_components_for_example_0})

    target_out = model(input)

    # THEN the output should be different for the first example (where it's routed to components)
    assert not torch.allclose(cm_routed_out[0], target_out[0])

    # but it should be the same for the second example (where it's not routed to components)
    assert torch.allclose(cm_routed_out[1], target_out[1])
