from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.metrics import schatten_loss
from spd.metrics.schatten_loss import SchattenLoss
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import LinearComponents
from spd.utils.module_utils import ModulePathInfo


class TestSchattenLoss:
    def test_basic_single_layer(self) -> None:
        """Test basic Schatten loss with a single layer and known values."""
        # Create components with known V and U values
        C, d_in, d_out = 2, 3, 4
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        # Set V and U to known values for easy calculation
        # V shape: [d_in, C] = [3, 2]
        # U shape: [C, d_out] = [2, 4]
        with torch.no_grad():
            components.V.copy_(torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))
            components.U.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]))

        # V_norms: ||V[:, 0]||^2 = 3*1^2 = 3, ||V[:, 1]||^2 = 3*2^2 = 12
        # U_norms: ||U[0, :]||^2 = 4*1^2 = 4, ||U[1, :]||^2 = 4*2^2 = 16
        # schatten_norms: [3+4, 12+16] = [7, 28]

        ci_upper_leaky = {"layer1": torch.tensor([[1.0, 1.0]])}
        # With pnorm=1 and ci_sum=[1, 1]:
        # loss = 1*7 + 1*28 = 35

        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=1.0,
        )
        expected = torch.tensor(35.0)
        assert torch.allclose(result, expected)

    def test_pnorm_effect(self) -> None:
        """Test that pnorm correctly affects CI weighting."""
        C, d_in, d_out = 2, 2, 2
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        with torch.no_grad():
            components.V.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            components.U.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        # V_norms: [1, 1], U_norms: [1, 1], schatten_norms: [2, 2]

        ci_upper_leaky = {"layer1": torch.tensor([[2.0, 3.0]])}
        # ci_sum = [2, 3]

        # With pnorm=1: loss = 2*2 + 3*2 = 10
        result_p1 = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=1.0,
        )
        assert torch.allclose(result_p1, torch.tensor(10.0))

        # With pnorm=2: loss = 4*2 + 9*2 = 26
        result_p2 = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=2.0,
        )
        assert torch.allclose(result_p2, torch.tensor(26.0))

    def test_batch_dimension_summing(self) -> None:
        """Test that CI values are correctly summed over batch dimensions."""
        C, d_in, d_out = 2, 2, 2
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        with torch.no_grad():
            components.V.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            components.U.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        # schatten_norms: [2, 2]

        # Batch size 2
        ci_upper_leaky = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        # ci_sum = [1+3, 2+4] = [4, 6]

        # With pnorm=1: loss = 4*2 + 6*2 = 20
        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=1.0,
        )
        assert torch.allclose(result, torch.tensor(20.0))

    def test_multiple_layers(self) -> None:
        """Test Schatten loss with multiple layers."""
        C, d_in, d_out = 2, 2, 2
        components1 = LinearComponents(C=C, d_in=d_in, d_out=d_out)
        components2 = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        with torch.no_grad():
            components1.V.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            components1.U.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            components2.V.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
            components2.U.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))

        # layer1 schatten_norms: [2, 2]
        # layer2 schatten_norms: [8, 8]

        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 1.0]]),
            "layer2": torch.tensor([[1.0, 1.0]]),
        }

        # With pnorm=1:
        # layer1 loss = 1*2 + 1*2 = 4
        # layer2 loss = 1*8 + 1*8 = 16
        # total = 20
        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components1, "layer2": components2},
            pnorm=1.0,
        )
        assert torch.allclose(result, torch.tensor(20.0))

    def test_zero_ci_gives_zero_loss(self) -> None:
        """Test that zero CI values give zero loss."""
        C, d_in, d_out = 2, 2, 2
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        ci_upper_leaky = {"layer1": torch.tensor([[0.0, 0.0]])}

        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=1.0,
        )
        assert torch.allclose(result, torch.tensor(0.0))

    def test_fractional_pnorm(self) -> None:
        """Test with fractional pnorm."""
        C, d_in, d_out = 1, 2, 2
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        with torch.no_grad():
            components.V.copy_(torch.tensor([[1.0], [1.0]]))
            components.U.copy_(torch.tensor([[1.0, 1.0]]))

        # V_norms: [2], U_norms: [2], schatten_norms: [4]

        ci_upper_leaky = {"layer1": torch.tensor([[4.0]])}
        # ci_sum = [4]

        # With pnorm=0.5: ci_weighted = 4^0.5 = 2
        # loss = 2 * 4 = 8
        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=0.5,
        )
        assert torch.allclose(result, torch.tensor(8.0))

    def test_happy_path_with_component_model(self) -> None:
        """Test Schatten loss in a realistic SPD workflow with ComponentModel.

        This test simulates a complete happy path:
        1. Create a simple target model
        2. Wrap it in a ComponentModel
        3. Set known V and U values
        4. Compute CI values
        5. Call schatten_loss
        6. Verify the output is correct
        """

        # Create a simple two-layer target model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 8, bias=False)
                self.layer2 = nn.Linear(8, 5, bias=False)

            @override
            def forward(self, x: Float[Tensor, "... 10"]) -> Float[Tensor, "... 5"]:
                x = self.layer1(x)
                x = self.layer2(x)
                return x

        target_model = SimpleModel()
        target_model.eval()
        target_model.requires_grad_(False)

        # Wrap in ComponentModel
        C = 4
        module_path_info = [
            ModulePathInfo(module_path="layer1", C=C),
            ModulePathInfo(module_path="layer2", C=C),
        ]

        comp_model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_fn_type="mlp",
            ci_fn_hidden_dims=[8],
            pretrained_model_output_attr=None,
            sigmoid_type="leaky_hard",
        )

        # Set known values for V and U for easy calculation
        with torch.no_grad():
            # Layer1: d_in=10, d_out=8, C=4
            # V shape: [10, 4], U shape: [4, 8]
            comp_model.components["layer1"].V.fill_(1.0)
            comp_model.components["layer1"].U.fill_(1.0)
            # V_norms: [10, 10, 10, 10], U_norms: [8, 8, 8, 8]
            # schatten_norms: [18, 18, 18, 18]

            # Layer2: d_in=8, d_out=5, C=4
            # V shape: [8, 4], U shape: [4, 5]
            comp_model.components["layer2"].V.fill_(2.0)
            comp_model.components["layer2"].U.fill_(2.0)
            # V_norms: [32, 32, 32, 32], U_norms: [20, 20, 20, 20]
            # schatten_norms: [52, 52, 52, 52]

        # Create CI values (batch size 2)
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            "layer2": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
        }

        # Test with pnorm=1
        # layer1:
        #   ci_sum = [2, 4, 6, 8]
        #   ci_weighted = [2, 4, 6, 8]
        #   loss = 2*18 + 4*18 + 6*18 + 8*18 = (2+4+6+8)*18 = 20*18 = 360
        # layer2:
        #   ci_sum = [1, 1, 1, 1]
        #   ci_weighted = [1, 1, 1, 1]
        #   loss = 1*52 + 1*52 + 1*52 + 1*52 = 4*52 = 208
        # total = 360 + 208 = 568

        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components=comp_model.components,
            pnorm=1.0,
        )
        expected = torch.tensor(568.0)
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

        # Test with pnorm=2
        # layer1:
        #   ci_sum = [2, 4, 6, 8]
        #   ci_weighted = [4, 16, 36, 64]
        #   loss = 4*18 + 16*18 + 36*18 + 64*18 = (4+16+36+64)*18 = 120*18 = 2160
        # layer2:
        #   ci_sum = [1, 1, 1, 1]
        #   ci_weighted = [1, 1, 1, 1]
        #   loss = 1*52 + 1*52 + 1*52 + 1*52 = 4*52 = 208
        # total = 2160 + 208 = 2368

        result_p2 = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components=comp_model.components,
            pnorm=2.0,
        )
        expected_p2 = torch.tensor(2368.0)
        assert torch.allclose(result_p2, expected_p2), f"Expected {expected_p2}, got {result_p2}"

        # Test with different CI values - sequence dimension (batch=2, seq=3, C=4)
        ci_upper_leaky_seq = {
            "layer1": torch.ones(2, 3, 4),
            "layer2": torch.ones(2, 3, 4) * 0.5,
        }

        # layer1:
        #   ci_sum = [6, 6, 6, 6] (2*3 for each component)
        #   ci_weighted = [6, 6, 6, 6]
        #   loss = 6*18 + 6*18 + 6*18 + 6*18 = 24*18 = 432
        # layer2:
        #   ci_sum = [3, 3, 3, 3] (2*3*0.5 for each component)
        #   ci_weighted = [3, 3, 3, 3]
        #   loss = 3*52 + 3*52 + 3*52 + 3*52 = 12*52 = 624
        # total = 432 + 624 = 1056

        result_seq = schatten_loss(
            ci_upper_leaky=ci_upper_leaky_seq,
            components=comp_model.components,
            pnorm=1.0,
        )
        expected_seq = torch.tensor(1056.0)
        assert torch.allclose(result_seq, expected_seq), (
            f"Expected {expected_seq}, got {result_seq}"
        )

    def test_schatten_loss_metric_class(self) -> None:
        """Test SchattenLoss Metric class with ComponentModel.

        This tests the full metric interface with update() and compute() methods,
        simulating how the metric would be used during SPD training.
        """

        # Create a simple target model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(6, 4, bias=False)

            @override
            def forward(self, x: Float[Tensor, "... 6"]) -> Float[Tensor, "... 4"]:
                return self.layer1(x)

        target_model = SimpleModel()
        target_model.eval()
        target_model.requires_grad_(False)

        # Wrap in ComponentModel
        C = 3
        comp_model = ComponentModel(
            target_model=target_model,
            module_path_info=[ModulePathInfo(module_path="layer1", C=C)],
            ci_fn_type="mlp",
            ci_fn_hidden_dims=[4],
            pretrained_model_output_attr=None,
            sigmoid_type="leaky_hard",
        )

        # Set known values for V and U
        with torch.no_grad():
            # Layer1: d_in=6, d_out=4, C=3
            # V shape: [6, 3], U shape: [3, 4]
            comp_model.components["layer1"].V.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                    ]
                )
            )
            comp_model.components["layer1"].U.copy_(
                torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
            )
            # V_norms: [6, 24, 54]
            # U_norms: [4, 16, 36]
            # schatten_norms: [10, 40, 90]

        # Create the metric
        device = "cpu"
        pnorm = 1.0
        metric = SchattenLoss(model=comp_model, device=device, pnorm=pnorm)

        # Simulate two batches
        # Batch 1
        ci_batch1 = CIOutputs(
            lower_leaky={"layer1": torch.tensor([[0.8, 0.6, 0.4]])},
            upper_leaky={"layer1": torch.tensor([[1.0, 2.0, 3.0]])},
            pre_sigmoid={"layer1": torch.tensor([[0.0, 0.0, 0.0]])},
        )
        metric.update(ci=ci_batch1)

        # Expected after batch 1:
        # ci_sum = [1, 2, 3]
        # ci_weighted = [1, 2, 3]
        # loss = 1*10 + 2*40 + 3*90 = 10 + 80 + 270 = 360

        # Batch 2
        ci_batch2 = CIOutputs(
            lower_leaky={"layer1": torch.tensor([[0.5, 0.5, 0.5]])},
            upper_leaky={"layer1": torch.tensor([[2.0, 1.0, 0.5]])},
            pre_sigmoid={"layer1": torch.tensor([[0.0, 0.0, 0.0]])},
        )
        metric.update(ci=ci_batch2)

        # Expected after batch 2:
        # ci_sum = [2, 1, 0.5]
        # ci_weighted = [2, 1, 0.5]
        # loss = 2*10 + 1*40 + 0.5*90 = 20 + 40 + 45 = 105
        # total = 360 + 105 = 465

        result = metric.compute()
        expected = torch.tensor(465.0)
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

        # Test with pnorm=2
        metric_p2 = SchattenLoss(model=comp_model, device=device, pnorm=2.0)

        # Batch 1 with pnorm=2
        metric_p2.update(ci=ci_batch1)
        # ci_sum = [1, 2, 3]
        # ci_weighted = [1, 4, 9]
        # loss = 1*10 + 4*40 + 9*90 = 10 + 160 + 810 = 980

        # Batch 2 with pnorm=2
        metric_p2.update(ci=ci_batch2)
        # ci_sum = [2, 1, 0.5]
        # ci_weighted = [4, 1, 0.25]
        # loss = 4*10 + 1*40 + 0.25*90 = 40 + 40 + 22.5 = 102.5
        # total = 980 + 102.5 = 1082.5

        result_p2 = metric_p2.compute()
        expected_p2 = torch.tensor(1082.5)
        assert torch.allclose(result_p2, expected_p2), f"Expected {expected_p2}, got {result_p2}"

    def test_3d_ci_tensor(self) -> None:
        """Test with 3D CI tensor (batch, seq, C)."""
        C, d_in, d_out = 2, 2, 2
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out)

        with torch.no_grad():
            components.V.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            components.U.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        # schatten_norms: [2, 2]

        # Shape: [batch=2, seq=3, C=2]
        ci_upper_leaky = {
            "layer1": torch.tensor(
                [
                    [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                    [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                ]
            )
        }
        # ci_sum over batch and seq: [6, 12]

        # With pnorm=1: loss = 6*2 + 12*2 = 36
        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components={"layer1": components},
            pnorm=1.0,
        )
        assert torch.allclose(result, torch.tensor(36.0))
