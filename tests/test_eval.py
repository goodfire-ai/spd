"""Tests for evaluation metrics and figures, particularly CIHistograms."""

from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from spd.configs import Config
from spd.eval import CIHistograms
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

    def test_max_batches_enforcement(
        self, mock_model: Mock, mock_config: Mock, sample_ci: dict[str, torch.Tensor]
    ):
        """Test that CIHistograms stops accumulating after max_batches."""
        max_batches = 3
        ci_hist = CIHistograms(mock_model, mock_config, max_batches=max_batches)

        # Create dummy batch and target_out
        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)

        # Watch more batches than max_batches
        for _ in range(max_batches + 2):
            ci_hist.watch_batch(batch, target_out, sample_ci)

        # Check that only max_batches were accumulated
        assert ci_hist.batches_seen == max_batches + 2  # Total batches seen
        assert len(ci_hist.causal_importances["layer1"]) == max_batches
        assert len(ci_hist.causal_importances["layer2"]) == max_batches

    def test_none_max_batches(
        self, mock_model: Mock, mock_config: Mock, sample_ci: dict[str, torch.Tensor]
    ):
        """Test unlimited batch accumulation when max_batches is None."""
        ci_hist = CIHistograms(mock_model, mock_config, max_batches=None)

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
