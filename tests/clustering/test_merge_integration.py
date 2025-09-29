"""Integration tests for the merge system with new samplers."""

import torch

from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig


class TestMergeIntegration:
    """Test the full merge iteration with different samplers."""

    def test_merge_with_range_sampler(self):
        """Test merge iteration with range sampler."""
        # Create test data
        n_samples = 100
        n_components = 10
        activations = torch.rand(n_samples, n_components)
        component_labels = [f"comp_{i}" for i in range(n_components)]

        # Configure with range sampler
        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=5,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.1},
            pop_component_prob=0,
            filter_dead_threshold=0.001,
        )

        # Run merge iteration
        history = merge_iteration(
            activations=activations,
            batch_id="test_merge_with_range_sampler",
            merge_config=config,
            component_labels=component_labels,
        )

        # Check results
        assert history is not None
        assert len(history.merges.k_groups) > 0
        assert history.merges.k_groups[0].item() == n_components
        # After iterations, should have fewer groups (merges reduce count)
        # Exact count depends on early stopping conditions
        assert history.merges.k_groups[-1].item() < n_components
        assert history.merges.k_groups[-1].item() >= 2  # Should stop before going below 2

    def test_merge_with_mcmc_sampler(self):
        """Test merge iteration with MCMC sampler."""
        # Create test data
        n_samples = 100
        n_components = 10
        activations = torch.rand(n_samples, n_components)
        component_labels = [f"comp_{i}" for i in range(n_components)]

        # Configure with MCMC sampler
        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=5,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 1.0},
            pop_component_prob=0,
            filter_dead_threshold=0.001,
        )

        # Run merge iteration
        history = merge_iteration(
            activations=activations,
            batch_id="test_merge_with_mcmc_sampler",
            merge_config=config,
            component_labels=component_labels,
        )

        # Check results
        assert history is not None
        assert len(history.merges.k_groups) > 0
        assert history.merges.k_groups[0].item() == n_components
        # Should have fewer groups after iterations
        assert history.merges.k_groups[-1].item() < n_components
        assert history.merges.k_groups[-1].item() >= 2

    def test_merge_with_popping(self):
        """Test merge iteration with component popping."""
        # Create test data
        n_samples = 100
        n_components = 15
        activations = torch.rand(n_samples, n_components)
        component_labels = [f"comp_{i}" for i in range(n_components)]

        # Configure with popping enabled
        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=10,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.05},
            pop_component_prob=0.3,  # 30% chance of popping
            filter_dead_threshold=0.001,
        )

        # Run merge iteration
        history = merge_iteration(
            activations=activations,
            batch_id="test_merge_with_popping",
            merge_config=config,
            component_labels=component_labels,
        )

        # Check results
        assert history is not None
        assert history.merges.k_groups[0].item() == n_components
        # Final group count depends on pops, but should be less than initial
        assert history.merges.k_groups[-1].item() < n_components

    def test_merge_comparison_samplers(self):
        """Compare behavior of different samplers with same data."""
        # Create test data with clear structure
        n_samples = 100
        n_components = 8
        activations = torch.rand(n_samples, n_components)

        # Make some components more active to create cost structure
        activations[:, 0] *= 2  # Component 0 is very active
        activations[:, 1] *= 0.1  # Component 1 is rarely active

        component_labels = [f"comp_{i}" for i in range(n_components)]

        # Run with range sampler (threshold=0 for deterministic minimum selection)
        config_range = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=3,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.0},  # Always select minimum
            pop_component_prob=0,
        )

        history_range = merge_iteration(
            activations=activations.clone(),
            batch_id="test_merge_comparison_samplers_range",
            merge_config=config_range,
            component_labels=component_labels.copy(),
        )

        # Run with MCMC sampler (low temperature for near-deterministic)
        config_mcmc = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=3,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 0.01},  # Very low temp
            pop_component_prob=0,
        )

        history_mcmc = merge_iteration(
            activations=activations.clone(),
            batch_id="test_merge_comparison_samplers_mcmc",
            merge_config=config_mcmc,
            component_labels=component_labels.copy(),
        )

        # Both should reduce groups from initial count
        assert history_range.merges.k_groups[-1].item() < n_components
        assert history_mcmc.merges.k_groups[-1].item() < n_components
        assert history_range.merges.k_groups[-1].item() >= 2
        assert history_mcmc.merges.k_groups[-1].item() >= 2

    def test_merge_with_small_components(self):
        """Test merge with very few components."""
        # Edge case: only 3 components
        n_samples = 50
        n_components = 3
        activations = torch.rand(n_samples, n_components)
        component_labels = [f"comp_{i}" for i in range(n_components)]

        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=1,  # Just one merge
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 2.0},
            pop_component_prob=0,
        )

        history = merge_iteration(
            activations=activations,
            batch_id="test_merge_with_small_components",
            merge_config=config,
            component_labels=component_labels,
        )

        # Should start with 3 components
        assert history.merges.k_groups[0].item() == 3
        # Early stopping may occur at 2 groups, so final count could be 2 or 3
        assert history.merges.k_groups[-1].item() >= 2
        assert history.merges.k_groups[-1].item() <= 3
