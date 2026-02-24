from unittest.mock import patch

import torch
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics import stochastic_recon_loss
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import Router
from tests.metrics.fixtures import make_one_layer_component_model


class TestStochasticReconLoss:
    def test_manual_calculation(self: object) -> None:
        """Test stochastic reconstruction with manual calculation.

        Mocks calc_stochastic_component_mask_info to use deterministic masks.
        """
        torch.manual_seed(42)

        fc_weight = torch.randn(2, 2, dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        V = model.components["fc"].V
        U = model.components["fc"].U

        batch = torch.randn(1, 2, dtype=torch.float32)
        target_out = torch.randn(1, 2, dtype=torch.float32)

        ci = {"fc": torch.tensor([[0.8]], dtype=torch.float32)}

        # Define deterministic masks for our samples
        # n_mask_samples=2, so we'll have 2 samples
        sample_masks = [
            torch.tensor([[0.9]], dtype=torch.float32),
            torch.tensor([[0.7]], dtype=torch.float32),
        ]

        # Mock calc_stochastic_component_mask_info to return our deterministic masks
        call_count = [0]

        def mock_calc_stochastic_component_mask_info(
            causal_importances: dict[str, Tensor],  # pyright: ignore[reportUnusedParameter]
            component_mask_sampling: SamplingType,  # pyright: ignore[reportUnusedParameter]
            router: Router,  # pyright: ignore[reportUnusedParameter]
            weight_deltas: dict[str, Tensor] | None,  # pyright: ignore[reportUnusedParameter]
        ) -> dict[str, ComponentsMaskInfo]:
            idx = call_count[0] % len(sample_masks)
            call_count[0] += 1
            masks = {"fc": sample_masks[idx]}

            return make_mask_infos(
                component_masks=masks,
                routing_masks="all",
                weight_deltas_and_masks=None,
            )

        with patch(
            "spd.metrics.stochastic_recon_loss.calc_stochastic_component_mask_info",
            side_effect=mock_calc_stochastic_component_mask_info,
        ):
            # Calculate expected loss manually
            sum_loss = 0.0
            n_examples = 0

            for mask in sample_masks:
                # Manually calculate forward pass: out = batch @ (V * mask @ U)
                masked_component = V * mask @ U
                out = batch @ masked_component
                loss = torch.nn.functional.mse_loss(out, target_out, reduction="sum")
                sum_loss += loss.item()
                n_examples += out.numel()

            expected_loss = sum_loss / n_examples

            # Calculate actual loss
            weight_deltas = model.calc_weight_deltas()
            actual_loss = stochastic_recon_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
            )

            assert torch.allclose(actual_loss, torch.tensor(expected_loss), rtol=1e-5), (
                f"Expected {expected_loss}, got {actual_loss}"
            )
