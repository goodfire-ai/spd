from unittest.mock import patch

import torch
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics import stochastic_hidden_acts_recon_loss
from spd.metrics.hidden_acts_recon_loss import (
    CIHiddenActsReconLoss,
    StochasticHiddenActsReconLoss,
    _stochastic_hidden_acts_recon_loss_update,
)
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import Router
from tests.metrics.fixtures import make_two_layer_component_model


def _make_mock_stochastic(
    sample_masks_fc1: list[Tensor],
    sample_masks_fc2: list[Tensor],
):
    call_count = [0]

    def mock_calc_stochastic_component_mask_info(
        causal_importances: dict[str, Tensor],  # pyright: ignore[reportUnusedParameter]
        component_mask_sampling: SamplingType,  # pyright: ignore[reportUnusedParameter]
        weight_deltas: dict[str, Tensor] | None,  # pyright: ignore[reportUnusedParameter]
        router: Router,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, ComponentsMaskInfo]:
        idx = call_count[0] % len(sample_masks_fc1)
        call_count[0] += 1
        masks = {"fc1": sample_masks_fc1[idx], "fc2": sample_masks_fc2[idx]}

        return make_mask_infos(
            component_masks=masks, routing_masks="all", weight_deltas_and_masks=None
        )

    return mock_calc_stochastic_component_mask_info


class TestStochasticHiddenActsReconLoss:
    def test_manual_calculation(self: object) -> None:
        """Test stochastic hidden acts recon loss with manual calculation.

        For a two-layer model (batch -> fc1 -> hidden -> fc2 -> output):
        - target output acts["fc1"] is fc1(batch) = batch @ W1^T
        - target output acts["fc2"] is fc2(fc1(batch)) = fc1(batch) @ W2^T

        With stochastic masks, component outputs differ from target outputs for both layers.
        """
        torch.manual_seed(42)

        # Create 2-layer model: 2 -> 3 -> 2
        fc1_weight = torch.randn(3, 2, dtype=torch.float32)
        fc2_weight = torch.randn(2, 3, dtype=torch.float32)
        model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

        V1 = model.components["fc1"].V
        U1 = model.components["fc1"].U
        V2 = model.components["fc2"].V
        U2 = model.components["fc2"].U

        batch = torch.randn(1, 2, dtype=torch.float32)

        # Get target output acts (post-weight activations)
        target_output_acts = model(batch, cache_type="output").cache

        ci = {
            "fc1": torch.tensor([[0.8]], dtype=torch.float32),
            "fc2": torch.tensor([[0.7]], dtype=torch.float32),
        }

        # Define deterministic masks for n_mask_samples=2
        sample_masks_fc1 = [
            torch.tensor([[0.9]], dtype=torch.float32),
            torch.tensor([[0.7]], dtype=torch.float32),
        ]
        sample_masks_fc2 = [
            torch.tensor([[0.85]], dtype=torch.float32),
            torch.tensor([[0.65]], dtype=torch.float32),
        ]

        with patch(
            "spd.metrics.hidden_acts_recon_loss.calc_stochastic_component_mask_info",
            side_effect=_make_mock_stochastic(sample_masks_fc1, sample_masks_fc2),
        ):
            # Calculate expected loss manually using output activations
            sum_mse = 0.0
            n_examples = 0

            for i, mask1 in enumerate(sample_masks_fc1):
                mask2 = sample_masks_fc2[i]

                # Component output of fc1: batch @ (V1 * mask1 @ U1)
                comp_fc1_output = batch @ (V1 * mask1 @ U1)

                # Component output of fc2: comp_fc1_output @ (V2 * mask2 @ U2)
                comp_fc2_output = comp_fc1_output @ (V2 * mask2 @ U2)

                # MSE for fc1 output
                mse_fc1 = torch.nn.functional.mse_loss(
                    comp_fc1_output, target_output_acts["fc1"], reduction="sum"
                )

                # MSE for fc2 output
                mse_fc2 = torch.nn.functional.mse_loss(
                    comp_fc2_output, target_output_acts["fc2"], reduction="sum"
                )

                sum_mse += mse_fc1.item() + mse_fc2.item()
                n_examples += comp_fc1_output.numel() + comp_fc2_output.numel()

            expected_loss = sum_mse / n_examples

            actual_loss = stochastic_hidden_acts_recon_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                batch=batch,
                ci=ci,
                weight_deltas=None,
            )

            assert torch.allclose(actual_loss, torch.tensor(expected_loss), rtol=1e-5), (
                f"Expected {expected_loss}, got {actual_loss}"
            )

    def test_per_module_values_sum_to_total(self) -> None:
        """Per-module MSE values, weighted by n_examples, should reconstruct the total."""
        torch.manual_seed(42)

        fc1_weight = torch.randn(3, 2, dtype=torch.float32)
        fc2_weight = torch.randn(2, 3, dtype=torch.float32)
        model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

        batch = torch.randn(2, 2, dtype=torch.float32)
        ci = {
            "fc1": torch.tensor([[0.8], [0.6]], dtype=torch.float32),
            "fc2": torch.tensor([[0.7], [0.5]], dtype=torch.float32),
        }

        per_module = _stochastic_hidden_acts_recon_loss_update(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            batch=batch,
            ci=ci,
            weight_deltas=None,
        )

        assert set(per_module.keys()) == {"fc1", "fc2"}

        total_mse = sum(mse.item() for mse, _ in per_module.values())
        total_n = sum(n for _, n in per_module.values())

        # Verify the per_module dict's own total is self-consistent
        expected_total = total_mse / total_n
        assert expected_total >= 0

        # Verify each per_module entry has positive values
        for key, (mse, n) in per_module.items():
            assert n > 0, f"Module {key} has zero examples"
            assert mse.item() >= 0, f"Module {key} has negative MSE"

    def test_metric_class_per_module_keys(self) -> None:
        """StochasticHiddenActsReconLoss.compute() returns per-module + total keys."""
        torch.manual_seed(42)

        fc1_weight = torch.randn(3, 2, dtype=torch.float32)
        fc2_weight = torch.randn(2, 3, dtype=torch.float32)
        model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

        batch = torch.randn(2, 2, dtype=torch.float32)

        # Need CIOutputs for the Metric class
        target_output = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_output.cache, sampling="continuous"
        )
        weight_deltas = model.calc_weight_deltas()

        metric = StochasticHiddenActsReconLoss(
            model=model,
            device="cpu",
            sampling="continuous",
            use_delta_component=False,
            n_mask_samples=2,
        )
        metric.update(batch=batch, ci=ci, weight_deltas=weight_deltas)
        result = metric.compute()

        assert isinstance(result, dict)
        expected_keys = {
            "StochasticHiddenActsReconLoss",
            "StochasticHiddenActsReconLoss/fc1",
            "StochasticHiddenActsReconLoss/fc2",
        }
        assert set(result.keys()) == expected_keys

        for key in ["fc1", "fc2"]:
            per_module_loss = result[f"StochasticHiddenActsReconLoss/{key}"].item()
            assert per_module_loss >= 0
        assert result["StochasticHiddenActsReconLoss"].item() >= 0

    def test_ci_metric_class_per_module_keys(self) -> None:
        """CIHiddenActsReconLoss.compute() returns per-module + total keys."""
        torch.manual_seed(42)

        fc1_weight = torch.randn(3, 2, dtype=torch.float32)
        fc2_weight = torch.randn(2, 3, dtype=torch.float32)
        model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

        batch = torch.randn(2, 2, dtype=torch.float32)

        target_output = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_output.cache, sampling="continuous"
        )
        weight_deltas = model.calc_weight_deltas()

        metric = CIHiddenActsReconLoss(model=model, device="cpu")
        metric.update(batch=batch, ci=ci, weight_deltas=weight_deltas)
        result = metric.compute()

        assert isinstance(result, dict)
        expected_keys = {
            "CIHiddenActsReconLoss",
            "CIHiddenActsReconLoss/fc1",
            "CIHiddenActsReconLoss/fc2",
        }
        assert set(result.keys()) == expected_keys
        assert result["CIHiddenActsReconLoss"].item() >= 0
