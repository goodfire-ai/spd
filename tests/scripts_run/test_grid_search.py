"""Tests for sweep functionality with discriminated lists and nested parameters.

All tests use realistic sweep parameters with discriminated lists (loss_metric_configs).
"""

import json

from spd.configs import Config, ImportanceMinimalityLossTrainConfig
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.tms.configs import TMSTaskConfig
from spd.utils.general_utils import load_config
from spd.utils.run_utils import apply_nested_updates, generate_grid_combinations


class TestGenerateGridCombinations:
    """Test generate_grid_combinations with realistic discriminated lists."""

    def test_simple_sweep_single_loss(self):
        """Test sweeping a single parameter in one loss config."""
        parameters = {
            "seed": {"values": [0, 1]},
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": {"values": [0.1, 0.2]},
                }
            ],
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4  # 2 seeds × 2 coeffs
        assert {
            "seed": 0,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
        } in combinations
        assert {
            "seed": 1,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.2,
        } in combinations

    def test_sweep_multiple_losses(self):
        """Test sweeping parameters across multiple loss configs."""
        parameters = {
            "seed": {"values": [0]},
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": {"values": [0.1, 0.2]},
                },
                {
                    "classname": "FaithfulnessLoss",
                    "coeff": {"values": [0.5]},
                },
            ],
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 2  # 2 × 1
        assert combinations[0] == {
            "seed": 0,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metric_configs.FaithfulnessLoss.coeff": 0.5,
        }
        assert combinations[1] == {
            "seed": 0,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.2,
            "loss_metric_configs.FaithfulnessLoss.coeff": 0.5,
        }

    def test_sweep_multiple_params_per_loss(self):
        """Test sweeping multiple parameters within same loss config."""
        parameters = {
            "seed": {"values": [0]},
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": {"values": [0.1, 0.2]},
                    "pnorm": {"values": [1.0, 2.0]},
                }
            ],
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4  # 2 × 2
        expected = [
            {
                "seed": 0,
                "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
                "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 1.0,
            },
            {
                "seed": 0,
                "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
                "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 2.0,
            },
            {
                "seed": 0,
                "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.2,
                "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 1.0,
            },
            {
                "seed": 0,
                "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.2,
                "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 2.0,
            },
        ]
        for exp in expected:
            assert exp in combinations

    def test_mixed_regular_and_discriminated_sweeps(self):
        """Test combining regular params with discriminated list sweeps."""
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": {"values": [0.001, 0.01]},
            "task_config": {"feature_probability": {"values": [0.05, 0.1]}},
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": {"values": [0.1, 0.2]},
                }
            ],
        }

        combinations = generate_grid_combinations(parameters)

        # 2 seeds × 2 lrs × 2 feature_probs × 2 coeffs = 16
        assert len(combinations) == 16
        assert {
            "seed": 0,
            "lr": 0.001,
            "task_config.feature_probability": 0.05,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
        } in combinations
        assert {
            "seed": 1,
            "lr": 0.01,
            "task_config.feature_probability": 0.1,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.2,
        } in combinations


class TestApplyNestedUpdates:
    """Test apply_nested_updates with realistic discriminated lists."""

    def test_update_existing_loss_config(self):
        """Test updating an existing loss config preserves other fields."""
        base = {
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

        updates = {
            "seed": 42,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 2.0,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "seed": 42,
            "lr": 0.001,
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.1,  # Updated
                    "pnorm": 2.0,  # Updated
                    "eps": 1e-12,  # Preserved
                },
                {
                    "classname": "FaithfulnessLoss",
                    "coeff": 1.0,  # Preserved
                },
            ],
        }

    def test_add_new_loss_config(self):
        """Test adding a new loss config not in base."""
        base = {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
            ],
        }

        updates = {
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 1.0,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "FaithfulnessLoss", "coeff": 1.0},  # Preserved
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.1,
                    "pnorm": 1.0,
                },  # Added
            ],
        }

    def test_multiple_losses_overlap(self):
        """Test complex scenario with overlapping losses."""
        base = {
            "seed": 0,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.5, "pnorm": 1.0, "eps": 1e-12},
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
                {"classname": "StochasticReconLoss", "coeff": 0.2},
            ],
        }

        updates = {
            "seed": 42,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 2.0,
            "loss_metric_configs.CIMaskedReconLoss.coeff": 0.3,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "seed": 42,
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.1,
                    "pnorm": 2.0,
                    "eps": 1e-12,
                },
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
                {"classname": "StochasticReconLoss", "coeff": 0.2},
                {"classname": "CIMaskedReconLoss", "coeff": 0.3},
            ],
        }

    def test_regular_nested_updates(self):
        """Test regular nested updates (non-discriminated)."""
        base = {"config": {"param1": 1, "param2": 2}, "other": 3}
        updates = {"config.param1": 10, "config.param3": 30}

        result = apply_nested_updates(base, updates)

        assert result == {"config": {"param1": 10, "param2": 2, "param3": 30}, "other": 3}

    def test_create_nested_structures(self):
        """Test creating new nested structures."""
        base = {"existing": 1}
        updates = {"new.nested.value": 42}

        result = apply_nested_updates(base, updates)

        assert result == {"existing": 1, "new": {"nested": {"value": 42}}}


class TestConfigIntegration:
    """Test end-to-end Config creation with realistic discriminated lists."""

    def test_tms_config_with_loss_sweep(self):
        """Test TMS config with loss_metric_configs sweep."""
        base_config = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["linear1"],
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.001,
                    "pnorm": 1.0,
                    "eps": 1e-12,
                },
                {
                    "classname": "FaithfulnessLoss",
                    "coeff": 1.0,
                },
            ],
            "output_loss_type": "mse",
            "lr": 0.001,
            "steps": 1000,
            "batch_size": 32,
            "train_log_freq": 100,
            "n_eval_steps": 100,
            "eval_batch_size": 32,
            "eval_freq": 100,
            "slow_eval_freq": 100,
            "ci_alive_threshold": 0.1,
            "n_examples_until_dead": 3200,
            "pretrained_model_class": "spd.experiments.tms.models.TMSModel",
            "task_config": {
                "task_name": "tms",
                "feature_probability": 0.05,
                "data_generation_type": "at_least_zero_active",
            },
        }

        updates = {
            "seed": 42,
            "task_config.feature_probability": 0.1,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.01,
        }

        updated_dict = apply_nested_updates(base_config, updates)
        config = Config(**updated_dict)

        assert config.seed == 42
        assert isinstance(config.task_config, TMSTaskConfig)
        assert config.task_config.feature_probability == 0.1
        assert config.loss_metric_configs[0].coeff == 0.01
        assert isinstance(config.loss_metric_configs[0], ImportanceMinimalityLossTrainConfig)
        assert config.loss_metric_configs[0].eps == 1e-12  # Preserved
        assert config.loss_metric_configs[1].coeff == 1.0  # Preserved

    def test_lm_config_with_loss_sweep(self):
        """Test LM config with loss_metric_configs sweep."""
        base_config = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["transformer"],
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.001,
                    "pnorm": 1.0,
                    "eps": 1e-12,
                }
            ],
            "output_loss_type": "kl",
            "lr": 0.001,
            "steps": 1000,
            "batch_size": 32,
            "train_log_freq": 100,
            "n_eval_steps": 100,
            "eval_batch_size": 32,
            "eval_freq": 100,
            "slow_eval_freq": 100,
            "ci_alive_threshold": 0.1,
            "n_examples_until_dead": 1638400,
            "pretrained_model_class": "transformers.LlamaForCausalLM",
            "task_config": {
                "task_name": "lm",
                "max_seq_len": 512,
                "buffer_size": 1000,
                "dataset_name": "test-dataset",
                "column_name": "text",
                "train_data_split": "train",
                "eval_data_split": "test",
            },
        }

        updates = {
            "task_config.max_seq_len": 256,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.01,
            "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 2.0,
        }

        updated_dict = apply_nested_updates(base_config, updates)
        config = Config(**updated_dict)

        assert isinstance(config.task_config, LMTaskConfig)
        assert config.task_config.max_seq_len == 256
        assert config.loss_metric_configs[0].coeff == 0.01
        assert isinstance(config.loss_metric_configs[0], ImportanceMinimalityLossTrainConfig)
        assert config.loss_metric_configs[0].pnorm == 2.0
        assert config.loss_metric_configs[0].eps == 1e-12  # Preserved

    def test_full_sweep_workflow(self):
        """Test complete sweep workflow: generate combinations → apply → create config."""
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": {"values": [0.001]},
            "task_config": {"feature_probability": {"values": [0.05, 0.1]}},
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": {"values": [0.001, 0.01]},
                }
            ],
        }

        base_config_dict = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["linear1"],
            "loss_metric_configs": [
                {
                    "classname": "ImportanceMinimalityLoss",
                    "coeff": 0.5,  # Will be overridden
                    "pnorm": 1.0,
                    "eps": 1e-12,
                }
            ],
            "output_loss_type": "mse",
            "lr": 0.01,  # Will be overridden
            "steps": 1000,
            "batch_size": 32,
            "train_log_freq": 100,
            "n_eval_steps": 100,
            "eval_batch_size": 32,
            "eval_freq": 100,
            "slow_eval_freq": 100,
            "ci_alive_threshold": 0.1,
            "n_examples_until_dead": 3200,
            "pretrained_model_class": "spd.experiments.tms.models.TMSModel",
            "task_config": {
                "task_name": "tms",
                "feature_probability": 0.2,  # Will be overridden
                "data_generation_type": "at_least_zero_active",
            },
        }

        combinations = generate_grid_combinations(parameters)
        assert len(combinations) == 8  # 2 × 1 × 2 × 2

        # Test each combination creates valid config
        for combo in combinations:
            updated_dict = apply_nested_updates(base_config_dict, combo)
            config = Config(**updated_dict)

            # Check overrides applied
            assert config.seed == combo["seed"]
            assert config.lr == combo["lr"]
            assert isinstance(config.task_config, TMSTaskConfig)
            assert (
                config.task_config.feature_probability == combo["task_config.feature_probability"]
            )
            assert (
                config.loss_metric_configs[0].coeff
                == combo["loss_metric_configs.ImportanceMinimalityLoss.coeff"]
            )

            # Check preserved values
            assert config.task_config.data_generation_type == "at_least_zero_active"
            assert isinstance(config.loss_metric_configs[0], ImportanceMinimalityLossTrainConfig)
            assert config.loss_metric_configs[0].pnorm == 1.0
            assert config.loss_metric_configs[0].eps == 1e-12

            # Verify JSON serialization round-trip
            json_str = f"json:{json.dumps(config.model_dump(mode='json'))}"
            reloaded_config = load_config(json_str, Config)
            assert reloaded_config.seed == config.seed
            assert (
                reloaded_config.loss_metric_configs[0].coeff == config.loss_metric_configs[0].coeff
            )
