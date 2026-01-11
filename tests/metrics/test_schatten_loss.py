import torch

from spd.metrics import schatten_loss
from spd.models.components import LinearComponents


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
