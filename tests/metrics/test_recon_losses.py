"""Sanity checks for stochastic, CI, and PGD reconstruction losses."""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from spd.configs import PGDConfig
from spd.metrics import ci_masked_recon_loss, pgd_recon_loss, stochastic_recon_loss
from spd.models.component_model import ComponentModel
from tests.metrics.fixtures import make_one_layer_component_model

ReconLossFn = Callable[[ComponentModel, Tensor, Tensor, dict[str, Tensor]], Tensor]


def _stochastic(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
) -> Tensor:
    return stochastic_recon_loss(
        model=model,
        sampling="continuous",
        n_mask_samples=4,
        output_loss_type="mse",
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=None,
    )


def _ci(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
) -> Tensor:
    return ci_masked_recon_loss(
        model=model,
        output_loss_type="mse",
        batch=batch,
        target_out=target_out,
        ci=ci,
    )


def _pgd(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
) -> Tensor:
    return pgd_recon_loss(
        model=model,
        batch=batch,
        target_out=target_out,
        output_loss_type="mse",
        ci=ci,
        weight_deltas=None,
        pgd_config=PGDConfig(
            init="random",
            step_size=0.1,
            n_steps=5,
            mask_scope="unique_per_datapoint",
        ),
    )


LOSS_FNS = [_stochastic, _ci, _pgd]


@pytest.mark.parametrize("loss_fn", LOSS_FNS, ids=["stochastic", "ci", "pgd"])
class TestReconLosses:
    def test_perfect_init_zero_recon(self, loss_fn: ReconLossFn) -> None:
        """V=W.T, U=I with CI=1 → all masks are 1 → output matches target → loss ≈ 0."""
        torch.manual_seed(42)
        d = 3
        weight = torch.randn(d, d)
        model = make_one_layer_component_model(weight=weight, C=d)

        target_weight = model.target_model.fc.weight.data
        with torch.no_grad():
            model.components["fc"].V.copy_(target_weight.T)
            model.components["fc"].U.copy_(torch.eye(d))

        batch = torch.randn(4, d)
        target_out = model.target_model(batch)
        ci = {"fc": torch.ones(4, d)}

        loss = loss_fn(model, batch, target_out, ci)
        assert loss < 1e-5, f"Expected ~0 loss with perfect init, got {loss}"

    def test_random_init_high_recon(self, loss_fn: ReconLossFn) -> None:
        """Random V and U should give substantially nonzero recon loss."""
        torch.manual_seed(42)
        d = 3
        weight = torch.randn(d, d)
        model = make_one_layer_component_model(weight=weight, C=d)

        batch = torch.randn(4, d)
        target_out = model.target_model(batch)
        ci = {"fc": torch.ones(4, d)}

        loss = loss_fn(model, batch, target_out, ci)
        assert loss > 0.01, f"Expected high loss with random init, got {loss}"


def test_manual_calculation() -> None:
    """Verify CI-masked recon loss matches a manual forward pass computation."""
    torch.manual_seed(42)

    fc_weight = torch.randn(2, 2)
    model = make_one_layer_component_model(weight=fc_weight)

    V = model.components["fc"].V
    U = model.components["fc"].U

    batch = torch.randn(1, 2)
    target_out = torch.randn(1, 2)
    ci = {"fc": torch.tensor([[0.8]])}

    # Manual: component_acts = batch @ V, masked = acts * ci, out = masked @ U
    out = (batch @ V * ci["fc"]) @ U
    expected_loss = torch.nn.functional.mse_loss(out, target_out, reduction="sum") / out.numel()

    actual_loss = ci_masked_recon_loss(
        model=model,
        output_loss_type="mse",
        batch=batch,
        target_out=target_out,
        ci=ci,
    )

    assert torch.allclose(actual_loss, expected_loss, rtol=1e-5), (
        f"Expected {expected_loss}, got {actual_loss}"
    )
