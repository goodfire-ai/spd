import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.losses import calc_faithfulness_loss
from spd.models.component_model import ComponentModel


class TinyLinearModel(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)


def _make_component_model(weight: Float[Tensor, " d_out d_in"]) -> ComponentModel:
    d_out, d_in = weight.shape
    target = TinyLinearModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(weight)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        target_module_patterns=["fc"],
        C=1,
        ci_fn_hidden_dims=[2],
        ci_fn_type="mlp",
        pretrained_model_output_attr=None,
    )

    return comp_model


def _zero_components_for_test(model: ComponentModel) -> None:
    with torch.no_grad():
        for cm in model.components.values():
            cm.V.zero_()
            cm.U.zero_()


class TestCalcWeightDeltas:
    def test_components_and_identity(self: object) -> None:
        # fc weight 2x3 with known values
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)
        _zero_components_for_test(model)

        deltas = model.calc_weight_deltas()

        assert set(deltas.keys()) == {"fc"}

        # components were zeroed, so delta equals original weight
        expected_fc = fc_weight
        assert torch.allclose(deltas["fc"], expected_fc)

    def test_components_nonzero(self: object) -> None:
        # TODO WRITE DESCRIPTION
        fc_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        deltas = model.calc_weight_deltas()
        assert set(deltas.keys()) == {"fc"}

        component = model.components["fc"]
        assert component is not None
        expected_fc = model.target_weight("fc") - component.weight
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
        model = _make_component_model(weight=fc_weight)
        _zero_components_for_test(model)
        deltas = model.calc_weight_deltas()

        # Expected: mean of squared entries across both matrices
        expected = fc_weight.square().sum() / fc_weight.numel()

        result = calc_faithfulness_loss(weight_deltas=deltas, device="cpu")
        assert torch.allclose(result, expected)
