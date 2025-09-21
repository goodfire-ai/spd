import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.losses import calc_faithfulness_loss, calc_weight_deltas
from spd.models.component_model import ComponentModel


class TinyLinearModel(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)


def _make_component_model(
    weight: Float[Tensor, " d_out d_in"],
    *,
    include_components: bool,
) -> ComponentModel:
    d_out, d_in = weight.shape
    target = TinyLinearModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(weight)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        target_module_patterns=["fc"] if include_components else [],
        C=1,
        gate_hidden_dims=[2],
        gate_type="mlp",
        pretrained_model_output_attr=None,
    )

    return comp_model


def _zero_components_for_test(model: ComponentModel, *, zero_components: bool) -> None:
    with torch.no_grad():
        if zero_components:
            for comp in model.components.values():
                comp.V.zero_()
                comp.U.zero_()


class TestCalcWeightDeltas:
    def test_components_zeroed(self: object) -> None:
        # fc weight 2x3 with known values
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight, include_components=True)
        _zero_components_for_test(model, zero_components=True)

        deltas = calc_weight_deltas(model=model)
        assert set(deltas.keys()) == {"fc"}
        expected_fc = fc_weight
        assert torch.allclose(deltas["fc"], expected_fc)

    def test_components_nonzero(self: object) -> None:
        # Non-identity case without zeroing components
        fc_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight, include_components=True)

        deltas = calc_weight_deltas(model=model)
        assert set(deltas.keys()) == {"fc"}

        expected_fc = model.get_original_weight("fc") - model.components["fc"].weight
        assert torch.allclose(deltas["fc"], expected_fc)


class TestCalcFaithfulnessLoss:
    def test_manual_weight_deltas_normalization(self: object) -> None:
        weight_deltas = {
            "a": torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float32),  # sum sq = 6
            "b": torch.tensor([[2.0, -2.0, 1.0]], dtype=torch.float32),  # sum sq = 9
        }
        # total sum sq = 15, total params = 4 + 3 = 7
        expected = torch.tensor(15.0 / 7.0)
        result = calc_faithfulness_loss(weight_deltas=weight_deltas, device="cpu")
        assert torch.allclose(result, expected)

    def test_with_model_weight_deltas(self: object) -> None:
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight, include_components=True)
        _zero_components_for_test(model, zero_components=True)
        deltas = calc_weight_deltas(model=model)

        # Expected: mean of squared entries of the single layer
        total_sq = fc_weight.square().sum()
        total_params = fc_weight.numel()
        expected = total_sq / total_params

        result = calc_faithfulness_loss(weight_deltas=deltas, device="cpu")
        assert torch.allclose(result, expected)
