"""Tests for SchattenLoss Metric class integration with evaluation."""

import torch

from spd.metrics.schatten_loss import SchattenLoss
from spd.models.component_model import CIOutputs
from tests.metrics.fixtures import make_one_layer_component_model, make_two_layer_component_model


class TestSchattenLossMetric:
    """Test the SchattenLoss metric class for evaluation integration."""

    def test_single_update_and_compute(self) -> None:
        """Test that a single update correctly accumulates and compute returns correct value."""
        # Create a simple component model
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        # Initialize metric
        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # Create CI outputs
        ci_upper_leaky = {"fc": torch.tensor([[2.0]], dtype=torch.float32)}
        ci = CIOutputs(
            lower_leaky={},
            upper_leaky=ci_upper_leaky,
            pre_sigmoid={},
        )

        # Calculate expected loss manually
        # V shape: [2, 1], U shape: [1, 2]
        component = model.components["fc"]
        V_norm = component.V.square().sum(dim=0)  # [1]
        U_norm = component.U.square().sum(dim=1)  # [1]
        schatten_norm = V_norm + U_norm  # [1]
        expected = (2.0**1.0 * schatten_norm).sum()

        # Update and compute
        metric.update(ci=ci)
        result = metric.compute()

        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiple_updates_accumulate(self) -> None:
        """Test that multiple updates correctly accumulate."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # First update
        ci1 = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[1.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )
        metric.update(ci=ci1)

        # Second update
        ci2 = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[2.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )
        metric.update(ci=ci2)

        # Third update
        ci3 = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[3.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )
        metric.update(ci=ci3)

        # Calculate expected accumulated loss
        component = model.components["fc"]
        V_norm = component.V.square().sum(dim=0)
        U_norm = component.U.square().sum(dim=1)
        schatten_norm = V_norm + U_norm

        expected = (
            (1.0 * schatten_norm).sum() + (2.0 * schatten_norm).sum() + (3.0 * schatten_norm).sum()
        )

        result = metric.compute()
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_zero_ci_accumulation(self) -> None:
        """Test edge case: zero CI values should give zero loss."""
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # Update with zero CI
        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[0.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )
        metric.update(ci=ci)
        metric.update(ci=ci)
        metric.update(ci=ci)

        result = metric.compute()
        assert torch.allclose(result, torch.tensor(0.0)), f"Expected 0.0, got {result}"

    def test_very_small_ci_values(self) -> None:
        """Test edge case: very small CI values."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # Very small CI values
        small_ci = 1e-8
        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[small_ci]], dtype=torch.float32)},
            pre_sigmoid={},
        )

        metric.update(ci=ci)
        result = metric.compute()

        # Should be non-zero but very small
        assert result > 0, "Result should be positive"
        assert result < 1e-6, "Result should be very small"

    def test_large_pnorm(self) -> None:
        """Test edge case: large pnorm values."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        # Test with large pnorm
        metric = SchattenLoss(model=model, device="cpu", pnorm=10.0)

        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[2.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )

        metric.update(ci=ci)
        result = metric.compute()

        # Expected: 2^10 * schatten_norm
        component = model.components["fc"]
        V_norm = component.V.square().sum(dim=0)
        U_norm = component.U.square().sum(dim=1)
        schatten_norm = V_norm + U_norm
        expected = (2.0**10.0) * schatten_norm.sum()

        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_fractional_pnorm(self) -> None:
        """Test with fractional pnorm values."""
        fc_weight = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=0.5)

        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[4.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )

        metric.update(ci=ci)
        result = metric.compute()

        # Expected: 4^0.5 * schatten_norm = 2 * schatten_norm
        component = model.components["fc"]
        V_norm = component.V.square().sum(dim=0)
        U_norm = component.U.square().sum(dim=1)
        schatten_norm = V_norm + U_norm
        expected = (4.0**0.5) * schatten_norm.sum()

        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_multiple_layers_accumulation(self) -> None:
        """Test accumulation with multiple layers."""
        weight1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        weight2 = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        model = make_two_layer_component_model(weight1=weight1, weight2=weight2)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # Update with CIs for both layers
        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={
                "fc1": torch.tensor([[1.0]], dtype=torch.float32),
                "fc2": torch.tensor([[2.0]], dtype=torch.float32),
            },
            pre_sigmoid={},
        )

        metric.update(ci=ci)
        result = metric.compute()

        # Calculate expected
        comp1 = model.components["fc1"]
        V_norm1 = comp1.V.square().sum(dim=0)
        U_norm1 = comp1.U.square().sum(dim=1)
        schatten_norm1 = V_norm1 + U_norm1

        comp2 = model.components["fc2"]
        V_norm2 = comp2.V.square().sum(dim=0)
        U_norm2 = comp2.U.square().sum(dim=1)
        schatten_norm2 = V_norm2 + U_norm2

        expected = (1.0 * schatten_norm1).sum() + (2.0 * schatten_norm2).sum()

        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_batch_dimension_summing_in_metric(self) -> None:
        """Test that batch dimensions are correctly summed before accumulation."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # CI with batch dimension [batch=2, C=1]
        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[1.0], [2.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )

        metric.update(ci=ci)
        result = metric.compute()

        # CI should sum over batch: [1+2] = [3]
        component = model.components["fc"]
        V_norm = component.V.square().sum(dim=0)
        U_norm = component.U.square().sum(dim=1)
        schatten_norm = V_norm + U_norm
        expected = (3.0 * schatten_norm).sum()

        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_3d_ci_tensor_accumulation(self) -> None:
        """Test with 3D CI tensor (batch, seq, C)."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # Shape: [batch=2, seq=3, C=1]
        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={
                "fc": torch.tensor(
                    [[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]], dtype=torch.float32
                )
            },
            pre_sigmoid={},
        )

        metric.update(ci=ci)
        result = metric.compute()

        # CI should sum over batch and seq: [3 + 6] = [9]
        component = model.components["fc"]
        V_norm = component.V.square().sum(dim=0)
        U_norm = component.U.square().sum(dim=1)
        schatten_norm = V_norm + U_norm
        expected = (9.0 * schatten_norm).sum()

        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_compute_is_idempotent(self) -> None:
        """Test that calling compute() multiple times returns the same value."""
        fc_weight = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # Some updates
        for _ in range(3):
            ci = CIOutputs(
                lower_leaky={},
                upper_leaky={"fc": torch.tensor([[1.0]], dtype=torch.float32)},
                pre_sigmoid={},
            )
            metric.update(ci=ci)

        result1 = metric.compute()
        result2 = metric.compute()
        result3 = metric.compute()

        assert torch.allclose(result1, result2), "compute() should be idempotent"
        assert torch.allclose(result2, result3), "compute() should be idempotent"

    def test_updates_after_compute_still_accumulate(self) -> None:
        """Test that updates after compute() still accumulate correctly."""
        fc_weight = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric = SchattenLoss(model=model, device="cpu", pnorm=1.0)

        # First batch of updates
        for _ in range(3):
            ci = CIOutputs(
                lower_leaky={},
                upper_leaky={"fc": torch.tensor([[1.0]], dtype=torch.float32)},
                pre_sigmoid={},
            )
            metric.update(ci=ci)

        intermediate_result = metric.compute()

        # More updates after compute
        for _ in range(2):
            ci = CIOutputs(
                lower_leaky={},
                upper_leaky={"fc": torch.tensor([[1.0]], dtype=torch.float32)},
                pre_sigmoid={},
            )
            metric.update(ci=ci)

        final_result = metric.compute()

        # Final result should be 5/3 times intermediate (5 updates vs 3)
        expected_ratio = 5.0 / 3.0
        actual_ratio = final_result / intermediate_result

        assert final_result > intermediate_result, (
            f"Final result {final_result} should be > intermediate {intermediate_result}"
        )
        assert torch.allclose(actual_ratio, torch.tensor(expected_ratio), rtol=1e-5), (
            f"Expected ratio {expected_ratio}, got {actual_ratio}"
        )

    def test_different_pnorms_give_different_results(self) -> None:
        """Test that different pnorms produce different accumulated losses."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        metric_p1 = SchattenLoss(model=model, device="cpu", pnorm=1.0)
        metric_p2 = SchattenLoss(model=model, device="cpu", pnorm=2.0)

        ci = CIOutputs(
            lower_leaky={},
            upper_leaky={"fc": torch.tensor([[2.0]], dtype=torch.float32)},
            pre_sigmoid={},
        )

        metric_p1.update(ci=ci)
        metric_p2.update(ci=ci)

        result_p1 = metric_p1.compute()
        result_p2 = metric_p2.compute()

        # p=2 should give 2^2 = 4, p=1 should give 2^1 = 2, so p2 result should be 2x p1
        assert result_p2 > result_p1, f"p=2 result {result_p2} should be > p=1 result {result_p1}"
        expected_ratio = 2.0
        actual_ratio = result_p2 / result_p1
        assert torch.allclose(actual_ratio, torch.tensor(expected_ratio), rtol=1e-5), (
            f"Expected ratio {expected_ratio}, got {actual_ratio}"
        )
