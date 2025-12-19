"""Tests for AliveComponentsTracker metric (single-rank)."""

import torch

from spd.metrics.alive_components import AliveComponentsTracker


def test_initialization():
    """Test that AliveComponentsTracker initializes correctly."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 5, "layer2": 5},
        device="cpu",
        n_examples_until_dead=100,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=2,
    )

    assert metric.n_examples_until_dead == 100
    assert metric.ci_alive_threshold == 0.1
    assert metric.n_batches_until_dead == 50
    assert "layer1" in metric.n_batches_since_fired
    assert "layer2" in metric.n_batches_since_fired
    assert metric.n_batches_since_fired["layer1"].shape == (5,)
    assert metric.n_batches_since_fired["layer2"].shape == (5,)


def test_update_counter_mechanics():
    """Test that firing resets counter to 0 and non-firing increments by 1."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 3},
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=1,
    )

    # Component 0 fires, components 1 and 2 don't
    ci = {"layer1": torch.tensor([0.2, 0.05, 0.08])}
    metric.update(ci=ci)

    assert metric.n_batches_since_fired["layer1"][0] == 0  # fired
    assert metric.n_batches_since_fired["layer1"][1] == 1  # didn't fire
    assert metric.n_batches_since_fired["layer1"][2] == 1  # didn't fire

    # No components fire - all should increment
    ci = {"layer1": torch.tensor([0.05, 0.08, 0.09])}
    metric.update(ci=ci)

    assert metric.n_batches_since_fired["layer1"][0] == 1
    assert metric.n_batches_since_fired["layer1"][1] == 2
    assert metric.n_batches_since_fired["layer1"][2] == 2

    # Component 1 fires
    ci = {"layer1": torch.tensor([0.05, 0.15, 0.09])}
    metric.update(ci=ci)

    assert metric.n_batches_since_fired["layer1"][0] == 2
    assert metric.n_batches_since_fired["layer1"][1] == 0  # reset
    assert metric.n_batches_since_fired["layer1"][2] == 3


def test_update_with_multidimensional_input():
    """Test that firing detection works with batch dimensions."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 3},
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=6,
    )

    # Shape: (batch=2, seq=3, C=3)
    # Component 0: fires in batch 0, token 0
    # Component 1: fires in batch 1, token 2
    # Component 2: never fires
    ci = {
        "layer1": torch.tensor(
            [
                [[0.2, 0.05, 0.08], [0.05, 0.08, 0.09], [0.05, 0.08, 0.09]],  # batch 0
                [[0.05, 0.08, 0.09], [0.05, 0.08, 0.09], [0.05, 0.12, 0.09]],  # batch 1
            ]
        )
    }

    metric.update(ci=ci)

    assert metric.n_batches_since_fired["layer1"][0] == 0  # fired
    assert metric.n_batches_since_fired["layer1"][1] == 0  # fired
    assert metric.n_batches_since_fired["layer1"][2] == 1  # didn't fire


def test_compute_alive_counts():
    """Test that compute() correctly counts alive components."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 4, "layer2": 4},
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=10,
    )

    # n_batches_until_dead = 50 // 10 = 5
    # Manually set counter values
    metric.n_batches_since_fired["layer1"] = torch.tensor([0, 3, 5, 10])
    metric.n_batches_since_fired["layer2"] = torch.tensor([4, 4, 6, 0])

    result = metric.compute()

    # layer1: components 0, 1 are alive (< 5)
    assert result["layer1"] == 2
    # layer2: components 0, 1, 3 are alive (< 5)
    assert result["layer2"] == 3


def test_multiple_modules():
    """Test tracking across multiple modules."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 3, "layer2": 3},
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=1,
    )

    ci = {
        "layer1": torch.tensor([0.2, 0.05, 0.08]),  # component 0 fires
        "layer2": torch.tensor([0.05, 0.12, 0.15]),  # components 1, 2 fire
    }

    metric.update(ci=ci)

    assert metric.n_batches_since_fired["layer1"][0] == 0
    assert metric.n_batches_since_fired["layer1"][1] == 1
    assert metric.n_batches_since_fired["layer1"][2] == 1

    assert metric.n_batches_since_fired["layer2"][0] == 1
    assert metric.n_batches_since_fired["layer2"][1] == 0
    assert metric.n_batches_since_fired["layer2"][2] == 0


def test_boundary_conditions():
    """Test boundary conditions for alive/dead determination."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 3},
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=10,
    )
    # n_batches_until_dead = 50 // 10 = 5
    # Test boundary: 4 < 5 (alive), 5 >= 5 (dead)
    metric.n_batches_since_fired["layer1"] = torch.tensor([4, 5, 6])

    result = metric.compute()
    assert result["layer1"] == 1  # only component 0


def test_threshold_boundary():
    """Test that the CI threshold is applied correctly."""
    metric = AliveComponentsTracker(
        module_to_c={"layer1": 3},
        device="cpu",
        n_examples_until_dead=50,
        ci_alive_threshold=0.1,
        global_n_examples_per_batch=10,
    )

    # Test boundary: 0.1 > 0.1 is False, so exactly 0.1 doesn't count as firing
    ci = {"layer1": torch.tensor([0.09, 0.1, 0.11])}
    metric.update(ci=ci)

    assert metric.n_batches_since_fired["layer1"][0] == 1  # 0.09 not > 0.1
    assert metric.n_batches_since_fired["layer1"][1] == 1  # 0.1 not > 0.1
    assert metric.n_batches_since_fired["layer1"][2] == 0  # 0.11 > 0.1
