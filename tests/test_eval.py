"""Tests for evaluation metrics and figures, particularly CIHistograms."""

from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from spd.configs import Config
from spd.eval import ActivationsAndInteractions, CIHistograms
from spd.models.component_model import ComponentModel


class TestCIHistograms:
    """Test suite for CIHistograms class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = Mock(spec=Config)
        config.ci_alive_threshold = 0.5
        config.sigmoid_type = "straight_through"
        return config

    @pytest.fixture
    def mock_model(self):
        """Create a mock ComponentModel."""
        model = Mock(spec=ComponentModel)
        model.C = 10  # Number of components
        model.components = {"layer1": Mock(), "layer2": Mock()}
        return model

    @pytest.fixture
    def sample_ci(self):
        """Create sample causal importance tensors."""
        return {
            "layer1": torch.randn(4, 8, 10),  # batch_size=4, seq_len=8, C=10
            "layer2": torch.randn(4, 8, 10),
        }

    def test_n_batches_accum_enforcement(
        self, mock_model: Mock, mock_config: Mock, sample_ci: dict[str, torch.Tensor]
    ):
        """Test that CIHistograms stops accumulating after n_batches_accum."""
        n_batches_accum = 3
        ci_hist = CIHistograms(mock_model, mock_config, n_batches_accum=n_batches_accum)

        # Create dummy batch and target_out
        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)

        # Watch more batches than n_batches_accum
        for _ in range(n_batches_accum + 2):
            ci_hist.watch_batch(batch, target_out, sample_ci)

        # Check that only n_batches_accum were accumulated
        assert ci_hist.batches_seen == n_batches_accum + 2  # Total batches seen
        assert len(ci_hist.causal_importances["layer1"]) == n_batches_accum
        assert len(ci_hist.causal_importances["layer2"]) == n_batches_accum

    def test_none_n_batches_accum(
        self, mock_model: Mock, mock_config: Mock, sample_ci: dict[str, torch.Tensor]
    ):
        """Test unlimited batch accumulation when n_batches_accum is None."""
        ci_hist = CIHistograms(mock_model, mock_config, n_batches_accum=None)

        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)

        # Watch many batches
        num_batches = 10
        for _ in range(num_batches):
            ci_hist.watch_batch(batch, target_out, sample_ci)

        # All batches should be accumulated
        assert ci_hist.batches_seen == num_batches
        assert len(ci_hist.causal_importances["layer1"]) == num_batches
        assert len(ci_hist.causal_importances["layer2"]) == num_batches

    def test_empty_compute(self, mock_model: Mock, mock_config: Mock):
        """Test compute() when no batches have been watched."""
        ci_hist = CIHistograms(mock_model, mock_config)

        with patch("spd.eval.plot_ci_values_histograms") as mock_plot:
            mock_plot.return_value = Image.new("RGB", (100, 100))

            # When no batches watched, causal_importances dict has empty lists
            # But we still need to mock the model.components to match the compute() logic
            result = ci_hist.compute()

            # Should still call plot with empty dict
            mock_plot.assert_called_once()
            combined_ci = mock_plot.call_args[1]["causal_importances"]

            # Dict will be empty since no batches were watched
            assert len(combined_ci) == 0
            assert "figures/causal_importance_values" in result


class TestActivationsAndInteractions:
    """Test suite for ActivationsAndInteractions class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = Mock(spec=Config)
        config.ci_alive_threshold = 0.5
        return config

    @pytest.fixture
    def mock_model(self):
        """Create a mock ComponentModel with components that have U and V matrices."""
        model = Mock(spec=ComponentModel)
        model.C = 10  # Number of components
        model.components = {
            "layer1": Mock(),
            "layer2": Mock(),
        }

        # Mock U and V matrices for each component
        for component in model.components.values():
            component.U.data = torch.randn(10, 20)  # C x d
            component.V.data = torch.randn(20, 10)  # d x C

        # Mock parameters to return a tensor with a device
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        model.parameters.return_value = [mock_param]

        return model

    @pytest.fixture
    def sample_ci(self):
        """Create sample causal importance tensors."""
        return {
            "layer1": torch.randn(4, 8, 10),  # batch_size=4, seq_len=8, C=10
            "layer2": torch.randn(4, 8, 10),
        }

    def test_individual_layer_plots(
        self, mock_model: Mock, mock_config: Mock, sample_ci: dict[str, torch.Tensor]
    ):
        """Test that ActivationsAndInteractions creates individual plots for each layer."""
        activations_and_interactions = ActivationsAndInteractions(mock_model, mock_config)

        # Create dummy batch and target_out
        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)

        # Watch a few batches to accumulate data
        for _ in range(3):
            activations_and_interactions.watch_batch(batch, target_out, sample_ci)

        with (
            patch(
                "spd.eval.plot_single_layer_component_co_activation_fractions"
            ) as mock_co_act_plot,
            patch(
                "spd.eval.plot_single_layer_component_abs_left_singular_vectors_geometric_interaction_strengths"
            ) as mock_geom_plot,
            patch(
                "spd.eval.plot_single_layer_geometric_interaction_strength_vs_coactivation"
            ) as mock_scatter_plot,
            patch(
                "spd.eval.plot_single_layer_geometric_interaction_strength_product_with_coactivation_fraction"
            ) as mock_product_plot,
        ):
            # Mock all plotting functions to return dummy images
            mock_co_act_plot.return_value = Image.new("RGB", (100, 100))
            mock_geom_plot.return_value = Image.new("RGB", (100, 100))
            mock_scatter_plot.return_value = Image.new("RGB", (100, 100))
            mock_product_plot.return_value = Image.new("RGB", (100, 100))

            result = activations_and_interactions.compute()

            # Check that we get individual plots for each layer
            expected_keys = [
                "figures/component_co_activation_fractions/layer1",
                "figures/component_co_activation_fractions/layer2",
                "figures/component_abs_left_singular_vectors_geometric_interaction_strengths/layer1",
                "figures/component_abs_left_singular_vectors_geometric_interaction_strengths/layer2",
                "figures/geometric_interaction_strength_vs_coactivation/layer1",
                "figures/geometric_interaction_strength_vs_coactivation/layer2",
                "figures/geometric_interaction_strength_product_with_coactivation_fraction/layer1",
                "figures/geometric_interaction_strength_product_with_coactivation_fraction/layer2",
            ]

            for key in expected_keys:
                assert key in result
                assert isinstance(result[key], Image.Image)

            # Check that plotting functions were called for each layer
            assert mock_co_act_plot.call_count == 2  # Once for each layer
            assert mock_geom_plot.call_count == 2
            assert mock_scatter_plot.call_count == 2
            assert mock_product_plot.call_count == 2
