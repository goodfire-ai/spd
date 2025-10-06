"""Test cases for sweeping over parameters in lists (e.g., loss_metric_configs).

Key constraint: All leaf values in sweep params must be {"values": [...]}.
Static values are represented as {"values": [single_value]}.
"""

from spd.scripts.run import generate_grid_combinations
from spd.utils.general_utils import apply_nested_updates


class TestListSweepBehavior:
    """Test expected behavior for list sweeps with structured parameters."""

    def test_simple_list_item_sweep(self):
        """Test sweeping a single parameter in one list item.

        All fields must have {values: [...]}, even the discriminator.
        """
        parameters = {
            "seed": {"values": [0]},
            "loss_metric_configs": [
                {
                    "classname": {"values": ["ImportanceMinimalityLoss"]},  # static, single value
                    "coeff": {"values": [0.1, 0.2]},  # sweep this
                }
            ],
        }

        combinations = generate_grid_combinations(parameters)

        # Expected: 2 combinations (one for each coeff value)
        assert len(combinations) == 2

        # Structure is preserved, values are unwrapped
        assert combinations[0] == {
            "seed": 0,
            "loss_metric_configs": [{"classname": "ImportanceMinimalityLoss", "coeff": 0.1}],
        }
        assert combinations[1] == {
            "seed": 0,
            "loss_metric_configs": [{"classname": "ImportanceMinimalityLoss", "coeff": 0.2}],
        }

    def test_multiple_list_items_sweep(self):
        """Test sweeping parameters across multiple list items.

        Each list item can have sweep params. We take cartesian product across all.
        """
        parameters = {
            "seed": {"values": [0]},
            "loss_metric_configs": [
                {
                    "classname": {"values": ["ImportanceMinimalityLoss"]},
                    "coeff": {"values": [0.1, 0.2]},
                },
                {
                    "classname": {"values": ["StochasticReconLayerwiseLoss"]},
                    "coeff": {"values": [0.5]},
                },
            ],
        }

        combinations = generate_grid_combinations(parameters)

        # Expected: 2 combinations (2 × 1)
        # Each combination should have BOTH loss configs
        assert len(combinations) == 2

        assert combinations[0] == {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.1},
                {"classname": "StochasticReconLayerwiseLoss", "coeff": 0.5},
            ],
        }
        assert combinations[1] == {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.2},
                {"classname": "StochasticReconLayerwiseLoss", "coeff": 0.5},
            ],
        }

    def test_multiple_params_per_list_item(self):
        """Test sweeping multiple parameters within same list item.

        Cartesian product applies within a single list item too.
        """
        parameters = {
            "seed": {"values": [0]},
            "loss_metric_configs": [
                {
                    "classname": {"values": ["ImportanceMinimalityLoss"]},
                    "coeff": {"values": [0.1, 0.2]},
                    "pnorm": {"values": [1.0, 2.0]},
                }
            ],
        }

        combinations = generate_grid_combinations(parameters)

        # Expected: 4 combinations (2 × 2)
        assert len(combinations) == 4

        expected_combos = [
            {
                "seed": 0,
                "loss_metric_configs": [
                    {"classname": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 1.0}
                ],
            },
            {
                "seed": 0,
                "loss_metric_configs": [
                    {"classname": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 2.0}
                ],
            },
            {
                "seed": 0,
                "loss_metric_configs": [
                    {"classname": "ImportanceMinimalityLoss", "coeff": 0.2, "pnorm": 1.0}
                ],
            },
            {
                "seed": 0,
                "loss_metric_configs": [
                    {"classname": "ImportanceMinimalityLoss", "coeff": 0.2, "pnorm": 2.0}
                ],
            },
        ]
        for expected in expected_combos:
            assert expected in combinations

    def test_apply_list_sweep_to_base_config(self):
        """Test applying list sweep combinations to a base config.

        We match list items by discriminator (classname) and merge fields.
        Items from base config that aren't in the combo are preserved.
        """
        base_config = {
            "seed": 0,
            "lr": 0.001,
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.5,
                    "pnorm": 1.0,
                    "eps": 1e-12,
                },
                {
                    "classname": "FaithfulnessLoss",
                    "coeff": 1.0,
                },
            ],
        }

        # This is output from generate_grid_combinations - no "values" keys
        # Combo only mentions ImportanceMinimalityLoss
        combo = {
            "seed": 42,
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.1,
                    "pnorm": 1.0,  # explicitly set in sweep params
                }
            ],
        }

        result = apply_nested_updates(base_config, combo)

        # FaithfulnessLoss is preserved since it's not in combo
        # ImportanceMinimalityLoss fields are merged: coeff and pnorm updated, eps preserved
        assert result == {
            "seed": 42,
            "lr": 0.001,
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.1,  # Updated from combo
                    "pnorm": 1.0,  # Updated from combo (same value)
                    "eps": 1e-12,  # Preserved from base (not in combo)
                },
                {
                    "classname": "FaithfulnessLoss",
                    "coeff": 1.0,  # Preserved from base (not in combo)
                },
            ],
        }

    def test_mixed_list_and_regular_sweep(self):
        """Test combining list sweeps with regular parameter sweeps.

        Cartesian product across ALL sweep parameters (inside and outside lists).
        """
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": {"values": [0.001, 0.01]},
            "loss_metric_configs": [
                {
                    "classname": {"values": ["ImportanceMinimalityLoss"]},
                    "coeff": {"values": [0.1, 0.2]},
                }
            ],
        }

        combinations = generate_grid_combinations(parameters)

        # Expected: 8 combinations (2 seeds × 2 lrs × 2 coeffs)
        assert len(combinations) == 8

        # Check a couple of combinations to verify structure
        assert {
            "seed": 0,
            "lr": 0.001,
            "loss_metric_configs": [{"classname": "ImportanceMinimalityLoss", "coeff": 0.1}],
        } in combinations
        assert {
            "seed": 1,
            "lr": 0.01,
            "loss_metric_configs": [{"classname": "ImportanceMinimalityLoss", "coeff": 0.2}],
        } in combinations

    def test_add_new_loss_config_not_in_base(self):
        """Test adding a new loss config that doesn't exist in base config.

        If sweep params reference a classname not in base, we should add it.
        """
        base_config = {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
            ],
        }

        # Combo introduces a new loss config
        combo = {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 1.0}
            ],
        }

        result = apply_nested_updates(base_config, combo)

        # Both loss configs should be present
        assert result == {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 1.0},  # Added
                {"classname": "FaithfulnessLoss", "coeff": 1.0},  # Preserved
            ],
        }

    def test_multiple_losses_in_sweep_and_base(self):
        """Test complex scenario with multiple losses in both sweep and base.

        Some overlap, some only in sweep, some only in base.
        """
        base_config = {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.5, "pnorm": 1.0, "eps": 1e-12},
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
                {"classname": "StochasticReconLoss", "coeff": 0.2},
            ],
        }

        combo = {
            "seed": 42,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 2.0},  # Updates
                {"classname": "CIMaskedReconLoss", "coeff": 0.3},  # New
            ],
        }

        result = apply_nested_updates(base_config, combo)

        # ImportanceMinimalityLoss updated, CIMaskedReconLoss added,
        # FaithfulnessLoss and StochasticReconLoss preserved
        assert result == {
            "seed": 42,
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.1,  # Updated
                    "pnorm": 2.0,  # Updated
                    "eps": 1e-12,  # Preserved from base
                },
                {"classname": "CIMaskedReconLoss", "coeff": 0.3},  # Added from combo
                {"classname": "FaithfulnessLoss", "coeff": 1.0},  # Preserved from base
                {"classname": "StochasticReconLoss", "coeff": 0.2},  # Preserved from base
            ],
        }
