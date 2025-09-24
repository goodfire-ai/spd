import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel


class TinyLinearModel(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)


def _make_component_model(
    weight: Float[Tensor, " d_out d_in"],
    *,
    include_components: bool,
    include_identity: bool,
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
        identity_module_patterns=["fc"] if include_identity else None,
    )

    return comp_model


def _zero_components_for_test(
    model: ComponentModel, *, zero_components: bool, zero_identity: bool
) -> None:
    with torch.no_grad():
        for cm in model.components_or_modules.values():
            if zero_components and cm.components is not None:
                cm.components.V.zero_()
                cm.components.U.zero_()
            if zero_identity and cm.identity_components is not None:
                cm.identity_components.V.zero_()
                cm.identity_components.U.zero_()


class TestCalcWeightDeltas:
    def test_components_and_identity(self: object) -> None:
        # fc weight 2x3 with known values
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(
            weight=fc_weight, include_components=True, include_identity=True
        )
        _zero_components_for_test(model, zero_components=True, zero_identity=True)

        deltas = model.calc_weight_deltas()

        assert set(deltas.keys()) == {"fc", "identity_fc"}

        # components were zeroed, so delta equals original weight
        expected_fc = fc_weight
        assert torch.allclose(deltas["fc"], expected_fc)

        # identity components were zeroed, so delta equals I - 0 = I
        d_in = fc_weight.shape[1]
        expected_I = torch.eye(d_in, dtype=fc_weight.dtype)
        assert torch.allclose(deltas["identity_fc"], expected_I)

    def test_identity_only(self: object) -> None:
        fc_weight = torch.tensor([[0.5, -1.5], [2.5, 3.5]], dtype=torch.float32)
        model = _make_component_model(
            weight=fc_weight, include_components=False, include_identity=True
        )
        _zero_components_for_test(model, zero_components=False, zero_identity=True)

        deltas = model.calc_weight_deltas()

        assert set(deltas.keys()) == {"identity_fc"}
        d_in = fc_weight.shape[1]
        expected_I = torch.eye(d_in, dtype=fc_weight.dtype)
        assert torch.allclose(deltas["identity_fc"], expected_I)

    def test_components_nonzero(self: object) -> None:
        # Non-identity case without zeroing components
        fc_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float32)
        model = _make_component_model(
            weight=fc_weight, include_components=True, include_identity=False
        )

        deltas = model.calc_weight_deltas()
        assert set(deltas.keys()) == {"fc"}

        cm = model.components_or_modules["fc"]
        assert cm.components is not None
        expected_fc = cm.original_weight - cm.components.weight
        assert torch.allclose(deltas["fc"], expected_fc)


# TODO: Write tests when we have functional version of faithfulness loss
# class TestCalcFaithfulnessLoss:
#     def test_manual_weight_deltas_normalization(self: object) -> None:
#         weight_deltas = {
#             "a": torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float32),  # sum sq = 6
#             "b": torch.tensor([[2.0, -2.0, 1.0]], dtype=torch.float32),  # sum sq = 9
#         }
#         # total sum sq = 15, total params = 4 + 3 = 7
#         expected = torch.tensor(15.0 / 7.0)
#         result = calc_faithfulness_loss(weight_deltas=weight_deltas, device="cpu")
#         assert torch.allclose(result, expected)

#     def test_with_model_weight_deltas(self: object) -> None:
#         fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
#         model = _make_component_model(
#             weight=fc_weight, include_components=True, include_identity=True
#         )
#         _zero_components_for_test(model, zero_components=True, zero_identity=True)
#         deltas = model.calc_weight_deltas()

#         # Expected: mean of squared entries across both matrices
#         d_in = fc_weight.shape[1]
#         total_sq = fc_weight.square().sum() + torch.eye(d_in, dtype=fc_weight.dtype).square().sum()
#         total_params = fc_weight.numel() + d_in * d_in
#         expected = total_sq / total_params

#         result = calc_faithfulness_loss(weight_deltas=deltas, device="cpu")
#         assert torch.allclose(result, expected)
