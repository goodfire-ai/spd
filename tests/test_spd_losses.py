from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossMetricConfigType,
    SchattenLossConfig,
    UniformKSubsetRoutingConfig,
    UnmaskedReconLossConfig,
)
from spd.losses import compute_total_loss
from spd.metrics import (
    ci_masked_recon_layerwise_loss,
    ci_masked_recon_loss,
    ci_masked_recon_subset_loss,
    faithfulness_loss,
    importance_minimality_loss,
    schatten_loss,
    stochastic_recon_layerwise_loss,
    stochastic_recon_loss,
    stochastic_recon_subset_loss,
)
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import ModulePathInfo


class TinyLinearModel(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _make_component_model(weight: Float[Tensor, "d_out d_in"]) -> ComponentModel:
    d_out, d_in = weight.shape
    target = TinyLinearModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(weight)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=[ModulePathInfo(module_path="fc", C=1)],
        ci_fn_hidden_dims=[2],
        ci_fn_type="mlp",
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
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
        result = faithfulness_loss(weight_deltas=weight_deltas)
        assert torch.allclose(result, expected)

    def test_with_model_weight_deltas(self: object) -> None:
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)
        _zero_components_for_test(model)
        deltas = model.calc_weight_deltas()

        # Expected: mean of squared entries across both matrices
        expected = fc_weight.square().sum() / fc_weight.numel()

        result = faithfulness_loss(weight_deltas=deltas)
        assert torch.allclose(result, expected)


class TestImportanceMinimalityLoss:
    def test_basic_l1_norm(self: object) -> None:
        # L1 norm: sum of absolute values (already positive with upper_leaky)
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            "layer2": torch.tensor([[0.5, 1.5]], dtype=torch.float32),
        }
        # With eps=0, p=1, pnorm_2=1, no annealing:
        # layer1: per_component_mean = [1, 2, 3], sum = 6
        # layer2: per_component_mean = [0.5, 1.5], sum = 2
        # total = 8
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        )
        expected = torch.tensor(8.0)
        assert torch.allclose(result, expected)

    def test_basic_l2_norm(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        }
        # L2: per_component_mean = [4, 9], sum = 13
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=2.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        )
        expected = torch.tensor(13.0)
        assert torch.allclose(result, expected)

    def test_epsilon_stability(self: object) -> None:
        # Verify epsilon prevents issues with zero values
        ci_upper_leaky = {
            "layer1": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }
        eps = 1e-6
        # With p=0.5, pnorm_2=1: per_component_mean = [(0+eps)^0.5, (1+eps)^0.5]
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=0.5,
            pnorm_2=1.0,
            eps=eps,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        )
        expected = (0.0 + eps) ** 0.5 + (1.0 + eps) ** 0.5
        assert torch.allclose(result, torch.tensor(expected))

    def test_p_annealing_before_start(self: object) -> None:
        # Before annealing starts, should use initial p
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.3,
            pnorm=2.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=0.5,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=1.0,
        )
        # Should use p=2: 2^2 = 4
        expected = torch.tensor(4.0)
        assert torch.allclose(result, expected)

    def test_p_annealing_during(self: object) -> None:
        # During annealing, should interpolate
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        # At 50% through annealing (0.25 between 0.0 and 0.5)
        # p should be: 2.0 + (1.0 - 2.0) * 0.5 = 1.5
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.25,
            pnorm=2.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=0.5,
        )
        # 2^1.5 = 2.828...
        expected = torch.tensor(2.0**1.5)
        assert torch.allclose(result, expected)

    def test_p_annealing_after_end(self: object) -> None:
        # After annealing ends, should use final p
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.9,
            pnorm=2.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=0.5,
        )
        # Should use p=1: 2^1 = 2
        expected = torch.tensor(2.0)
        assert torch.allclose(result, expected)

    def test_no_annealing_when_final_p_none(self: object) -> None:
        # When p_anneal_final_p is None, should always use initial p
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.9,
            pnorm=2.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=0.5,
        )
        # Should use p=2: 2^2 = 4
        expected = torch.tensor(4.0)
        assert torch.allclose(result, expected)

    def test_multiple_layers_aggregation(self: object) -> None:
        # Test that losses from multiple layers are correctly summed
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "layer2": torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        }
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            pnorm_2=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        )
        # layer1: per_component_mean = [1, 1], sum = 2
        # layer2: per_component_mean = [2, 2], sum = 4
        # total = 6
        expected = torch.tensor(6.0)
        assert torch.allclose(result, expected)


class TestCIMaskedReconLoss:
    def test_mse_loss_basic(self: object) -> None:
        # Test basic MSE reconstruction loss
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        # Input and target
        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # CI values (will be used to mask components)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}  # Full component weight

        result = ci_masked_recon_loss(
            model=model,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
        )

        # Since we're using a simple identity-like weight, and CI is 1,
        # the reconstruction should be close (not exact due to component decomposition)
        assert result >= 0.0

    def test_kl_loss_basic(self: object) -> None:
        # Test basic KL divergence loss
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        # Use log-probs for KL
        target_out = torch.nn.functional.log_softmax(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32), dim=-1
        )

        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        result = ci_masked_recon_loss(
            model=model,
            output_loss_type="kl",
            batch=batch,
            target_out=target_out,
            ci=ci,
        )

        assert result >= 0.0

    def test_different_ci_values_produce_different_losses(self: object) -> None:
        # Test that different CI values produce different reconstruction losses
        fc_weight = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        target_out = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

        ci_full = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        ci_half = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}

        loss_full = ci_masked_recon_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci_full
        )
        loss_half = ci_masked_recon_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci_half
        )

        # Different CI values should produce different losses
        assert loss_full != loss_half


class TestCIMaskedReconLayerwiseLoss:
    def test_layerwise_basic(self: object) -> None:
        # Test layerwise reconstruction - each layer is evaluated separately
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        result = ci_masked_recon_layerwise_loss(
            model=model,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
        )

        # Layerwise should produce a valid loss
        assert result >= 0.0

    def test_layerwise_vs_all_layer(self: object) -> None:
        # Layerwise should differ from all-layer when there are multiple layers
        # For a single layer, they should be similar
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        loss_all = ci_masked_recon_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci
        )
        loss_layerwise = ci_masked_recon_layerwise_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci
        )

        # For single layer, results should be the same
        assert torch.allclose(loss_all, loss_layerwise, rtol=1e-4)


class TestCIMaskedReconSubsetLoss:
    def test_subset_basic(self: object) -> None:
        # Test subset routing reconstruction
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        result = ci_masked_recon_subset_loss(
            model=model,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            routing=UniformKSubsetRoutingConfig(),
        )

        # Subset routing should produce a valid loss
        assert result >= 0.0

    def test_subset_stochastic_behavior(self: object) -> None:
        # Subset routing has randomness, so repeated calls may differ
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        # Run multiple times
        losses = [
            ci_masked_recon_subset_loss(
                model=model,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                routing=UniformKSubsetRoutingConfig(),
            )
            for _ in range(3)
        ]

        # All should be valid losses (>= 0)
        assert all(loss >= 0.0 for loss in losses)


class TestStochasticReconLoss:
    def test_continuous_sampling_basic(self: object) -> None:
        # Test stochastic reconstruction with continuous sampling
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        assert result >= 0.0

    def test_binomial_sampling_basic(self: object) -> None:
        # Test stochastic reconstruction with binomial sampling
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_loss(
            model=model,
            sampling="binomial",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        assert result >= 0.0

    def test_multiple_mask_samples(self: object) -> None:
        # Test that using more mask samples produces valid results
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        # Test with different numbers of samples
        for n_samples in [1, 3, 5]:
            result = stochastic_recon_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=n_samples,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
            )
            assert result >= 0.0

    def test_with_and_without_delta_component(self: object) -> None:
        # Test both with and without delta component
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        loss_with_delta = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        loss_without_delta = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
        )

        # Both should be valid
        assert loss_with_delta >= 0.0
        assert loss_without_delta >= 0.0


class TestStochasticReconLayerwiseLoss:
    def test_layerwise_stochastic_basic(self: object) -> None:
        # Test layerwise stochastic reconstruction
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_layerwise_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=2,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        assert result >= 0.0

    def test_layerwise_multiple_samples(self: object) -> None:
        # Test with different numbers of mask samples
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.8]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        for n_samples in [1, 2, 3]:
            result = stochastic_recon_layerwise_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=n_samples,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
            )
            assert result >= 0.0


class TestStochasticReconSubsetLoss:
    def test_subset_stochastic_basic(self: object) -> None:
        # Test subset stochastic reconstruction
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_subset_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
            routing=UniformKSubsetRoutingConfig(),
        )

        assert result >= 0.0

    def test_subset_with_binomial_sampling(self: object) -> None:
        # Test subset with binomial sampling
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.7]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_subset_loss(
            model=model,
            sampling="binomial",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
            routing=UniformKSubsetRoutingConfig(),
        )

        assert result >= 0.0

    def test_subset_stochastic_variability(self: object) -> None:
        # Test that stochastic subset routing produces valid results across runs
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        losses = [
            stochastic_recon_subset_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
                routing=UniformKSubsetRoutingConfig(),
            )
            for _ in range(3)
        ]

        # All should be valid
        assert all(loss >= 0.0 for loss in losses)


class TestSchattenLoss:
    def test_basic_computation(self: object) -> None:
        # Test basic Schatten loss computation
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        # CI values
        ci_upper_leaky = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}

        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components=model.components,
            pnorm=1.0,
        )

        # Should be a scalar tensor
        assert result.dim() == 0
        assert result >= 0.0

    def test_different_pnorms(self: object) -> None:
        # Test with different p-norms
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        ci_upper_leaky = {"fc": torch.tensor([[0.8]], dtype=torch.float32)}

        for pnorm in [0.5, 1.0, 2.0]:
            result = schatten_loss(
                ci_upper_leaky=ci_upper_leaky,
                components=model.components,
                pnorm=pnorm,
            )
            assert result >= 0.0

    def test_zero_ci_produces_zero_loss(self: object) -> None:
        # Test that zero CI produces zero loss
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        ci_upper_leaky = {"fc": torch.tensor([[0.0]], dtype=torch.float32)}

        result = schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            components=model.components,
            pnorm=1.0,
        )

        # Zero CI should produce zero loss
        assert torch.allclose(result, torch.tensor(0.0))


class TestComputeTotalLossWithSchattenLoss:
    def test_schatten_loss_in_total_loss(self: object) -> None:
        # Test that SchattenLoss is correctly integrated into compute_total_loss
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        # Create batch
        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # Forward pass with caching to get pre-weight activations
        output_with_cache = model(batch, cache_type="input")
        assert hasattr(output_with_cache, "cache"), "Expected OutputWithCache"
        target_out = output_with_cache.output

        # Compute CI outputs
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            detach_inputs=False,
            sampling="continuous",
        )

        # Get weight deltas
        weight_deltas = model.calc_weight_deltas()

        # Create loss configs including SchattenLoss
        loss_configs: list[LossMetricConfigType] = [
            SchattenLossConfig(pnorm=1.0, coeff=0.1),
        ]

        # Compute total loss
        total_loss, terms = compute_total_loss(
            loss_metric_configs=loss_configs,
            model=model,
            batch=batch,
            ci=ci,
            target_out=target_out,
            weight_deltas=weight_deltas,
            pre_weight_acts=output_with_cache.cache,
            current_frac_of_training=0.0,
            sampling="continuous",
            use_delta_component=True,
            n_mask_samples=1,
            output_loss_type="mse",
        )

        # Verify total loss is valid
        assert total_loss >= 0.0
        assert "loss/SchattenLoss" in terms
        assert terms["loss/SchattenLoss"] >= 0.0
        assert "loss/total" in terms
        # Total should be weighted by coeff (0.1)
        assert torch.allclose(total_loss, torch.tensor(terms["loss/total"]))

    def test_schatten_loss_with_multiple_losses(self: object) -> None:
        # Test SchattenLoss combined with other losses
        fc_weight = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        output_with_cache = model(batch, cache_type="input")
        target_out = output_with_cache.output
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            detach_inputs=False,
            sampling="continuous",
        )
        weight_deltas = model.calc_weight_deltas()

        # Multiple loss configs
        loss_configs: list[LossMetricConfigType] = [
            FaithfulnessLossConfig(coeff=0.5),
            SchattenLossConfig(pnorm=1.0, coeff=0.1),
            ImportanceMinimalityLossConfig(pnorm=1.0, coeff=0.2),
        ]

        total_loss, terms = compute_total_loss(
            loss_metric_configs=loss_configs,
            model=model,
            batch=batch,
            ci=ci,
            target_out=target_out,
            weight_deltas=weight_deltas,
            pre_weight_acts=output_with_cache.cache,
            current_frac_of_training=0.0,
            sampling="continuous",
            use_delta_component=True,
            n_mask_samples=1,
            output_loss_type="mse",
        )

        # Verify all losses are present
        assert "loss/FaithfulnessLoss" in terms
        assert "loss/SchattenLoss" in terms
        assert "loss/ImportanceMinimalityLoss" in terms
        assert "loss/total" in terms

        # Verify total is sum of weighted individual losses
        expected_total = (
            0.5 * terms["loss/FaithfulnessLoss"]
            + 0.1 * terms["loss/SchattenLoss"]
            + 0.2 * terms["loss/ImportanceMinimalityLoss"]
        )
        assert torch.allclose(total_loss, torch.tensor(expected_total), rtol=1e-5)

    def test_schatten_loss_with_reconstruction_loss(self: object) -> None:
        # Test SchattenLoss combined with reconstruction loss
        fc_weight = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        output_with_cache = model(batch, cache_type="input")
        target_out = output_with_cache.output
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            detach_inputs=False,
            sampling="continuous",
        )
        weight_deltas = model.calc_weight_deltas()

        loss_configs: list[LossMetricConfigType] = [
            UnmaskedReconLossConfig(coeff=1.0),
            SchattenLossConfig(pnorm=2.0, coeff=0.05),
        ]

        total_loss, terms = compute_total_loss(
            loss_metric_configs=loss_configs,
            model=model,
            batch=batch,
            ci=ci,
            target_out=target_out,
            weight_deltas=weight_deltas,
            pre_weight_acts=output_with_cache.cache,
            current_frac_of_training=0.0,
            sampling="continuous",
            use_delta_component=True,
            n_mask_samples=1,
            output_loss_type="mse",
        )

        # Verify both losses are present and valid
        assert "loss/UnmaskedReconLoss" in terms
        assert "loss/SchattenLoss" in terms
        assert terms["loss/UnmaskedReconLoss"] >= 0.0
        assert terms["loss/SchattenLoss"] >= 0.0

        # Verify total matches expected weighted sum
        expected_total = 1.0 * terms["loss/UnmaskedReconLoss"] + 0.05 * terms["loss/SchattenLoss"]
        assert torch.allclose(total_loss, torch.tensor(expected_total), rtol=1e-5)

    def test_schatten_loss_different_pnorms_in_total_loss(self: object) -> None:
        # Test different p-norm values in SchattenLoss
        fc_weight = torch.tensor([[1.5, 0.5], [0.5, 1.5]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        output_with_cache = model(batch, cache_type="input")
        target_out = output_with_cache.output
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            detach_inputs=False,
            sampling="continuous",
        )
        weight_deltas = model.calc_weight_deltas()

        for pnorm in [0.5, 1.0, 1.5, 2.0]:
            loss_configs: list[LossMetricConfigType] = [SchattenLossConfig(pnorm=pnorm, coeff=0.1)]

            total_loss, terms = compute_total_loss(
                loss_metric_configs=loss_configs,
                model=model,
                batch=batch,
                ci=ci,
                target_out=target_out,
                weight_deltas=weight_deltas,
                pre_weight_acts=output_with_cache.cache,
                current_frac_of_training=0.0,
                sampling="continuous",
                use_delta_component=True,
                n_mask_samples=1,
                output_loss_type="mse",
            )

            # All should produce valid losses
            assert total_loss >= 0.0
            assert terms["loss/SchattenLoss"] >= 0.0

    def test_schatten_loss_e2e_training_scenario(self: object) -> None:
        # End-to-end test simulating a training scenario like in run_spd.py
        # This tests the happy path: model forward, CI calculation, loss computation
        fc_weight = torch.tensor([[1.5, 0.3], [0.3, 1.5]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        # Simulate multiple training steps
        for step in range(3):
            batch = torch.randn(2, 2)  # Random batch

            # Step 1: Forward pass with caching (like in run_spd.py line 261)
            output_with_cache = model(batch, cache_type="input")
            target_out = output_with_cache.output

            # Step 2: Calculate causal importances (like in run_spd.py line 263-267)
            ci = model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                detach_inputs=False,
                sampling="continuous",
            )

            # Step 3: Get weight deltas (like in run_spd.py line 251)
            weight_deltas = model.calc_weight_deltas()

            # Step 4: Compute total loss with multiple loss types including Schatten
            # (like in run_spd.py line 271-284)
            loss_configs: list[LossMetricConfigType] = [
                FaithfulnessLossConfig(coeff=0.5),
                ImportanceMinimalityLossConfig(pnorm=1.0, coeff=0.2),
                SchattenLossConfig(pnorm=1.0, coeff=0.1),
                UnmaskedReconLossConfig(coeff=1.0),
            ]

            total_loss, terms = compute_total_loss(
                loss_metric_configs=loss_configs,
                model=model,
                batch=batch,
                ci=ci,
                target_out=target_out,
                weight_deltas=weight_deltas,
                pre_weight_acts=output_with_cache.cache,
                current_frac_of_training=step / 3.0,
                sampling="continuous",
                use_delta_component=True,
                n_mask_samples=2,
                output_loss_type="mse",
            )

            # Verify all losses are computed correctly
            assert total_loss >= 0.0
            assert "loss/FaithfulnessLoss" in terms
            assert "loss/ImportanceMinimalityLoss" in terms
            assert "loss/SchattenLoss" in terms
            assert "loss/UnmaskedReconLoss" in terms
            assert "loss/total" in terms

            # Verify all individual losses are valid
            for loss_name, loss_value in terms.items():
                if loss_name != "loss/total":
                    assert loss_value >= 0.0, f"{loss_name} should be non-negative"

            # Verify total matches weighted sum
            expected_total = (
                0.5 * terms["loss/FaithfulnessLoss"]
                + 0.2 * terms["loss/ImportanceMinimalityLoss"]
                + 0.1 * terms["loss/SchattenLoss"]
                + 1.0 * terms["loss/UnmaskedReconLoss"]
            )
            assert torch.allclose(total_loss, torch.tensor(expected_total), rtol=1e-5)

            # Verify loss is differentiable (can compute gradients)
            total_loss.backward()

            # Check that gradients were computed for component parameters
            for param in model.components["fc"].parameters():
                assert param.grad is not None, "Components should have gradients"

            # Reset gradients for next iteration
            model.zero_grad()
