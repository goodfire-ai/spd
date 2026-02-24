from unittest.mock import patch

import torch
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics import stochastic_hidden_acts_recon_loss
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import Router
from tests.metrics.fixtures import make_two_layer_component_model


class TestStochasticHiddenActsReconLoss:
    def test_manual_calculation(self: object) -> None:
        """Test stochastic hidden acts recon loss with manual calculation.

        For a two-layer model (batch -> fc1 -> hidden -> fc2 -> output):
        - pre_weight_acts["fc1"] is the batch input (always same, MSE = 0)
        - pre_weight_acts["fc2"] is the hidden activation (differs with stochastic masks)

        This structure provides both a sanity check and meaningful test of the metric.
        """
        torch.manual_seed(42)

        # Create 2-layer model: 2 -> 3 -> 2
        fc1_weight = torch.randn(3, 2, dtype=torch.float32)
        fc2_weight = torch.randn(2, 3, dtype=torch.float32)
        model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

        V1 = model.components["fc1"].V
        U1 = model.components["fc1"].U

        batch = torch.randn(1, 2, dtype=torch.float32)

        # Get target pre_weight_acts (activations before each weight matrix)
        # fc1: input is batch
        # fc2: input is output of fc1
        target_pre_weight_acts = model(batch, cache_type="input").cache

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

        # Mock calc_stochastic_component_mask_info to return our deterministic masks
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

        with patch(
            "spd.metrics.stochastic_hidden_acts_recon_loss.calc_stochastic_component_mask_info",
            side_effect=mock_calc_stochastic_component_mask_info,
        ):
            # Calculate expected loss manually
            sum_mse = 0.0
            n_examples = 0

            for mask1 in sample_masks_fc1:
                # Stochastic forward pass for fc1
                # pre_weight_acts["fc1"] is always batch (same as target)
                stoch_fc1_input = batch

                # pre_weight_acts["fc2"] is output of masked fc1
                stoch_fc2_input = batch @ (V1 * mask1 @ U1)

                # MSE for fc1 input (should be 0 - good sanity check!)
                mse_fc1 = torch.nn.functional.mse_loss(
                    stoch_fc1_input, target_pre_weight_acts["fc1"], reduction="sum"
                )
                assert mse_fc1.item() == 0.0, f"MSE for fc1 input should be 0, got {mse_fc1.item()}"

                # MSE for fc2 input (the actual meaningful comparison)
                mse_fc2 = torch.nn.functional.mse_loss(
                    stoch_fc2_input, target_pre_weight_acts["fc2"], reduction="sum"
                )

                sum_mse += mse_fc1.item() + mse_fc2.item()
                n_examples += stoch_fc1_input.numel() + stoch_fc2_input.numel()

            expected_loss = sum_mse / n_examples

            actual_loss = stochastic_hidden_acts_recon_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                batch=batch,
                pre_weight_acts=target_pre_weight_acts,
                ci=ci,
                weight_deltas=model.calc_weight_deltas(),
            )

            assert torch.allclose(actual_loss, torch.tensor(expected_loss), rtol=1e-5), (
                f"Expected {expected_loss}, got {actual_loss}"
            )
