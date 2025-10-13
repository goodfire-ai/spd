"""Tests for filter_dead_components function in activations.py"""

import pytest
import torch
from torch import Tensor

from spd.clustering.activations import FilteredActivations, filter_dead_components
from spd.clustering.consts import SubComponentKey, SubComponentLabel


@pytest.mark.parametrize(
    "max_values,threshold,expected_alive_indices",
    [
        # No filtering when threshold is 0
        ([0.1, 0.2, 0.3], 0.0, [0, 1, 2]),
        # Filter all when all below threshold
        ([0.005, 0.003, 0.004], 0.01, []),
        # Filter some components
        ([0.0, 0.02, 0.0, 0.03, 0.0], 0.01, [1, 3]),
        # Boundary cases: at threshold is kept
        ([0.009, 0.01, 0.011], 0.01, [1, 2]),
        # High threshold filters everything
        ([0.1, 0.2, 0.3], 2.0, []),
        # Negative threshold filters nothing
        ([0.1, 0.2, 0.3], -0.01, [0, 1, 2]),
        # Single component above threshold
        ([0.5], 0.01, [0]),
    ],
)
def test_filter_dead_components_thresholds(
    max_values: list[float],
    threshold: float,
    expected_alive_indices: list[int],
) -> None:
    """Test filtering with various max values and thresholds."""
    n_steps: int = 10
    n_components: int = len(max_values)

    activations: Tensor
    labels: list[SubComponentKey]
    if n_components == 0:
        activations = torch.zeros(n_steps, 0)
        labels = []
    else:
        activations = torch.zeros(n_steps, n_components)
        # Set max values in first row
        for i, val in enumerate(max_values):
            activations[0, i] = val
        labels = [SubComponentKey(module="comp", index=i) for i in range(n_components)]

    result: FilteredActivations = filter_dead_components(
        activations=activations, subcomponent_keys=labels, filter_dead_threshold=threshold
    )

    assert [comp.label for comp in result.subcomponent_keys] == [
        f"comp:{i}" for i in expected_alive_indices
    ]
    assert result.n_alive == len(expected_alive_indices)
    assert result.n_dead == n_components - len(expected_alive_indices)
    assert result.activations.shape == (n_steps, len(expected_alive_indices))

    # Check dead components labels
    if threshold <= 0 or all(v >= threshold for v in max_values):
        # No filtering occurred
        assert result.dead_subcomponent_keys is None or result.dead_subcomponent_keys == []
    else:
        dead_indices: list[int] = [
            i for i in range(n_components) if i not in expected_alive_indices
        ]
        expected_dead: list[SubComponentLabel] = [
            SubComponentKey(module="comp", index=i).label for i in dead_indices
        ]
        assert result.dead_subcomponent_keys is not None
        assert set(result.dead_subcomponent_keys) == set(expected_dead)


@pytest.mark.parametrize(
    "step_locations,threshold",
    [
        # Max at different steps
        ([0, 5, 9], 0.01),
        # All at same step
        ([0, 0, 0], 0.01),
        # Random steps
        ([3, 7, 1, 8], 0.05),
    ],
)
def test_max_across_steps(step_locations: list[int], threshold: float) -> None:
    """Verify that filter_dead_components correctly finds the maximum activation
    across ALL time steps for each component, not just looking at a single step.

    This test creates components where the maximum activation occurs at different
    time steps, ensuring the function scans the entire temporal dimension."""
    n_steps: int = 10
    n_components: int = len(step_locations)
    activations: Tensor = torch.zeros(n_steps, n_components)

    # Set values above threshold at specified steps
    for i, step in enumerate(step_locations):
        activations[step, i] = threshold + 0.01

    labels: list[SubComponentKey] = [
        SubComponentKey(module="comp", index=i) for i in range(n_components)
    ]

    result: FilteredActivations = filter_dead_components(
        activations=activations, subcomponent_keys=labels, filter_dead_threshold=threshold
    )

    # All components should be alive since their max is above threshold
    assert result.n_alive == n_components
    assert result.n_dead == 0
    assert result.subcomponent_keys == labels


@pytest.mark.parametrize("threshold", [0.001, 0.01, 0.1, 0.5])
def test_linear_gradient_thresholds(threshold: float) -> None:
    """Test with linearly spaced activation values."""
    n_steps: int = 10
    n_components: int = 10
    activations: Tensor = torch.zeros(n_steps, n_components)

    # Create linearly spaced max values: 0, 0.1, 0.2, ..., 0.9
    for i in range(n_components):
        activations[0, i] = i * 0.1

    labels: list[SubComponentKey] = [
        SubComponentKey(module="comp", index=i) for i in range(n_components)
    ]

    result: FilteredActivations = filter_dead_components(
        activations=activations, subcomponent_keys=labels, filter_dead_threshold=threshold
    )

    # Count how many components should be alive
    expected_alive: int = sum(i * 0.1 >= threshold for i in range(n_components))

    assert result.n_alive == expected_alive
    assert result.n_dead == n_components - expected_alive
