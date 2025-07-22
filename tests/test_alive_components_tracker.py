"""Tests for AliveComponentsTracker."""

import torch

from spd.utils.alive_components_tracker import AliveComponentsTracker


def test_watch_batch_single_example():
    """Test watching a single example batch."""
    module_names = ["layer1", "layer2"]
    C = 5
    n_examples_until_dead = 10
    device = torch.device("cpu")
    ci_alive_threshold = 0.1

    tracker = AliveComponentsTracker(
        module_names=module_names,
        C=C,
        n_examples_until_dead=n_examples_until_dead,
        device=device,
        ci_alive_threshold=ci_alive_threshold,
    )

    # Create importance values where some components fire (> threshold)
    # layer1: components 0, 2 fire; 1, 3, 4 don't
    # layer2: components 1, 3, 4 fire; 0, 2 don't
    importance_vals = {
        "layer1": torch.tensor([0.2, 0.05, 0.15, 0.0, 0.08], device=device),
        "layer2": torch.tensor([0.05, 0.12, 0.08, 0.2, 0.11], device=device),
    }

    tracker.watch_batch(importance_vals)

    # Check that firing components reset to 0
    assert tracker.examples_since_fired_C["layer1"][0] == 0
    assert tracker.examples_since_fired_C["layer1"][2] == 0
    assert tracker.examples_since_fired_C["layer2"][1] == 0
    assert tracker.examples_since_fired_C["layer2"][3] == 0
    assert tracker.examples_since_fired_C["layer2"][4] == 0

    # Check that non-firing components increment by 1
    assert tracker.examples_since_fired_C["layer1"][1] == 1
    assert tracker.examples_since_fired_C["layer1"][3] == 1
    assert tracker.examples_since_fired_C["layer1"][4] == 1
    assert tracker.examples_since_fired_C["layer2"][0] == 1
    assert tracker.examples_since_fired_C["layer2"][2] == 1


def test_watch_batch_multiple_examples():
    """Test watching a batch with multiple examples."""
    module_names = ["layer1"]
    C = 3
    n_examples_until_dead = 10
    device = torch.device("cpu")
    ci_alive_threshold = 0.1

    tracker = AliveComponentsTracker(
        module_names=module_names,
        C=C,
        n_examples_until_dead=n_examples_until_dead,
        device=device,
        ci_alive_threshold=ci_alive_threshold,
    )

    # Batch of 4 examples, shape (4, 3)
    # Component 0: fires in examples 0 and 2
    # Component 1: never fires
    # Component 2: fires in example 3
    importance_vals = {
        "layer1": torch.tensor(
            [
                [0.2, 0.05, 0.08],  # Example 0: component 0 fires
                [0.05, 0.08, 0.09],  # Example 1: no components fire
                [0.15, 0.0, 0.05],  # Example 2: component 0 fires
                [0.08, 0.09, 0.12],  # Example 3: component 2 fires
            ],
            device=device,
        )
    }

    tracker.watch_batch(importance_vals)

    # Component 0 fired, so should be 0
    assert tracker.examples_since_fired_C["layer1"][0] == 0
    # Component 1 never fired, so should be 4 (batch size)
    assert tracker.examples_since_fired_C["layer1"][1] == 4
    # Component 2 fired, so should be 0
    assert tracker.examples_since_fired_C["layer1"][2] == 0


def test_n_alive():
    """Test counting alive components."""
    module_names = ["layer1", "layer2"]
    C = 4
    n_examples_until_dead = 5
    device = torch.device("cpu")
    ci_alive_threshold = 0.1

    tracker = AliveComponentsTracker(
        module_names=module_names,
        C=C,
        n_examples_until_dead=n_examples_until_dead,
        device=device,
        ci_alive_threshold=ci_alive_threshold,
    )

    # Manually set examples since fired
    tracker.examples_since_fired_C["layer1"] = torch.tensor([0, 3, 5, 10], device=device)
    tracker.examples_since_fired_C["layer2"] = torch.tensor([4, 4, 6, 0], device=device)

    n_alive = tracker.n_alive()

    # layer1: components 0 (0 < 5) and 1 (3 < 5) are alive
    assert n_alive["layer1"] == 2
    # layer2: components 0 (4 < 5), 1 (4 < 5), and 3 (0 < 5) are alive
    assert n_alive["layer2"] == 3


def test_is_alive():
    """Test getting alive boolean masks."""
    module_names = ["layer1"]
    C = 4
    n_examples_until_dead = 5
    device = torch.device("cpu")
    ci_alive_threshold = 0.1

    tracker = AliveComponentsTracker(
        module_names=module_names,
        C=C,
        n_examples_until_dead=n_examples_until_dead,
        device=device,
        ci_alive_threshold=ci_alive_threshold,
    )

    # Manually set examples since fired
    tracker.examples_since_fired_C["layer1"] = torch.tensor([0, 3, 5, 10], device=device)

    is_alive = tracker.is_alive()

    expected = torch.tensor([True, True, False, False], device=device)
    assert torch.equal(is_alive["layer1"], expected)


def test_sequence_dimensions():
    """Test with sequence dimensions (e.g., for language models)."""
    module_names = ["embedding"]
    C = 3
    n_examples_until_dead = 100
    device = torch.device("cpu")
    ci_alive_threshold = 0.1

    tracker = AliveComponentsTracker(
        module_names=module_names,
        C=C,
        n_examples_until_dead=n_examples_until_dead,
        device=device,
        ci_alive_threshold=ci_alive_threshold,
    )

    # Batch size 2, sequence length 5, components 3
    # Shape: (2, 5, 3)
    importance_vals = {
        "embedding": torch.tensor(
            [
                # Batch item 0
                [
                    [0.2, 0.05, 0.08],  # Token 0: component 0 fires
                    [0.05, 0.08, 0.09],  # Token 1: no components fire
                    [0.05, 0.12, 0.08],  # Token 2: component 1 fires
                    [0.08, 0.09, 0.05],  # Token 3: no components fire
                    [0.08, 0.09, 0.15],  # Token 4: component 2 fires
                ],
                # Batch item 1
                [
                    [0.05, 0.08, 0.09],  # Token 0: no components fire
                    [0.15, 0.05, 0.08],  # Token 1: component 0 fires
                    [0.05, 0.08, 0.09],  # Token 2: no components fire
                    [0.08, 0.12, 0.05],  # Token 3: component 1 fires
                    [0.08, 0.09, 0.05],  # Token 4: no components fire
                ],
            ],
            device=device,
        )
    }

    tracker.watch_batch(importance_vals)

    # Component 0 fired in both batch items, so should be 0
    assert tracker.examples_since_fired_C["embedding"][0] == 0
    # Component 1 fired in both batch items, so should be 0
    assert tracker.examples_since_fired_C["embedding"][1] == 0
    # Component 2 fired only in batch item 0, so should be 0
    assert tracker.examples_since_fired_C["embedding"][2] == 0

    # Make a new batch where components 1 and 2 fire, but not 0
    # - c0: doesn't fire in either batch
    # - c1: fires only in batch item 1
    # - c2: fires in both batch items
    importance_vals = {
        "embedding": torch.tensor(
            [
                # Batch item 0
                [
                    [0.0, 0.0, 0.0],  # Token 0: component 0 fires
                    [0.0, 0.0, 0.0],  # Token 1: no components fire
                    [0.0, 0.0, 0.0],  # Token 2: component 1 fires
                    [0.0, 0.0, 0.0],  # Token 3: no components fire
                    [0.0, 0.0, 0.2],  # Token 4: component 2 fires
                ],
                # Batch item 1
                [
                    [0.0, 0.0, 0.0],  # Token 0: no components fire
                    [0.0, 0.0, 0.0],  # Token 1: component 0 fires
                    [0.0, 0.0, 0.0],  # Token 2: no components fire
                    [0.0, 0.0, 0.0],  # Token 3: component 1 fires
                    [0.0, 0.2, 0.2],  # Token 4: no components fire
                ],
            ],
            device=device,
        )
    }

    tracker.watch_batch(importance_vals)

    # All components should increment by batch_size * seq_len = 2 * 5 = 10
    assert tracker.examples_since_fired_C["embedding"][0] == 10
    assert tracker.examples_since_fired_C["embedding"][1] == 0
    assert tracker.examples_since_fired_C["embedding"][2] == 0
