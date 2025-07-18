"""Tests for sweep functionality with nested parameters."""

import json
from typing import Any

from spd.configs import Config, LMTaskConfig, TMSTaskConfig
from spd.scripts.run import generate_grid_combinations
from spd.utils.general_utils import apply_nested_updates, load_config


class TestGenerateGridCombinations:
    """Test the generate_grid_combinations function with various nesting scenarios."""

    def test_simple_parameters(self):
        """Test generation with simple (non-nested) parameters."""
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": {"values": [0.001, 0.01]},
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4
        assert {"seed": 0, "lr": 0.001} in combinations
        assert {"seed": 0, "lr": 0.01} in combinations
        assert {"seed": 1, "lr": 0.001} in combinations
        assert {"seed": 1, "lr": 0.01} in combinations

    def test_single_level_nesting(self):
        """Test generation with single level of nesting."""
        parameters = {
            "seed": {"values": [0, 1]},
            "task_config": {"feature_probability": {"values": [0.05, 0.1]}},
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4
        assert {"seed": 0, "task_config.feature_probability": 0.05} in combinations
        assert {"seed": 0, "task_config.feature_probability": 0.1} in combinations
        assert {"seed": 1, "task_config.feature_probability": 0.05} in combinations
        assert {"seed": 1, "task_config.feature_probability": 0.1} in combinations

    def test_multiple_nested_parameters(self):
        """Test generation with multiple nested parameters in same group."""
        parameters = {
            "seed": {"values": [0]},
            "task_config": {
                "max_seq_len": {"values": [128, 256]},
                "buffer_size": {"values": [1000, 2000]},
            },
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4
        expected_combos = [
            {"seed": 0, "task_config.max_seq_len": 128, "task_config.buffer_size": 1000},
            {"seed": 0, "task_config.max_seq_len": 128, "task_config.buffer_size": 2000},
            {"seed": 0, "task_config.max_seq_len": 256, "task_config.buffer_size": 1000},
            {"seed": 0, "task_config.max_seq_len": 256, "task_config.buffer_size": 2000},
        ]
        for combo in expected_combos:
            assert combo in combinations

    def test_deep_nesting(self):
        """Test generation with multiple levels of nesting."""
        parameters = {"level1": {"level2": {"level3": {"values": [1, 2]}}}}

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 2
        assert {"level1.level2.level3": 1} in combinations
        assert {"level1.level2.level3": 2} in combinations

    def test_mixed_nesting_styles(self):
        """Test generation with mixed nesting styles."""
        parameters = {
            "simple": {"values": [1]},
            "nested_param": {"param1": {"values": [2]}},
            "direct_nested": {"sub_param": {"values": [3]}},
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 1
        assert combinations[0] == {
            "simple": 1,
            "nested_param.param1": 2,
            "direct_nested.sub_param": 3,
        }


class TestApplyNestedUpdates:
    """Test the apply_nested_updates function."""

    def test_simple_updates(self):
        """Test applying simple (non-nested) updates."""
        base = {"a": 1, "b": 2}
        updates = {"a": 10, "c": 3}

        result = apply_nested_updates(base, updates)

        assert result == {"a": 10, "b": 2, "c": 3}
        assert base == {"a": 1, "b": 2}  # Original unchanged

    def test_single_level_nested_updates(self):
        """Test applying single level nested updates."""
        base: dict[str, Any] = {"config": {"param1": 1, "param2": 2}, "other": 3}
        updates = {"config.param1": 10, "config.param3": 30}

        result = apply_nested_updates(base, updates)

        assert result == {"config": {"param1": 10, "param2": 2, "param3": 30}, "other": 3}
        assert base["config"]["param1"] == 1  # Original unchanged

    def test_deep_nested_updates(self):
        """Test applying deeply nested updates."""
        base = {"level1": {"level2": {"level3": {"value": 1}}}}
        updates = {"level1.level2.level3.value": 100}

        result = apply_nested_updates(base, updates)

        assert result["level1"]["level2"]["level3"]["value"] == 100
        assert base["level1"]["level2"]["level3"]["value"] == 1  # Original unchanged

    def test_create_missing_nested_structures(self):
        """Test that missing nested structures are created."""
        base = {"existing": 1}
        updates = {"new.nested.value": 42}

        result = apply_nested_updates(base, updates)

        assert result == {"existing": 1, "new": {"nested": {"value": 42}}}

    def test_mixed_updates(self):
        """Test applying mix of simple and nested updates."""
        base = {"simple": 1, "nested": {"a": 2, "b": 3}}
        updates = {"simple": 10, "nested.a": 20, "nested.c": 30, "new.param": 40}

        result = apply_nested_updates(base, updates)

        assert result == {"simple": 10, "nested": {"a": 20, "b": 3, "c": 30}, "new": {"param": 40}}


class TestConfigIntegration:
    """Test that Config objects can be created with nested updates."""

    def test_tms_config_with_nested_updates(self):
        """Test creating TMS config with nested task_config updates."""
        base_config = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["linear1"],
            "importance_minimality_coeff": 0.001,
            "pnorm": 1.0,
            "output_loss_type": "mse",
            "lr": 0.001,
            "steps": 1000,
            "batch_size": 32,
            "n_eval_steps": 100,
            "print_freq": 100,
            "pretrained_model_class": "spd.experiments.tms.models.TMSModel",
            "task_config": {
                "task_name": "tms",
                "feature_probability": 0.05,
                "data_generation_type": "at_least_zero_active",
            },
        }

        updates = {"seed": 42, "task_config.feature_probability": 0.1, "lr": 0.01}

        updated_dict = apply_nested_updates(base_config, updates)
        config = Config(**updated_dict)

        assert config.seed == 42
        assert config.lr == 0.01
        assert isinstance(config.task_config, TMSTaskConfig)
        assert config.task_config.feature_probability == 0.1
        assert config.task_config.data_generation_type == "at_least_zero_active"

    def test_lm_config_with_nested_updates(self):
        """Test creating LM config with nested task_config updates."""
        base_config = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["transformer"],
            "importance_minimality_coeff": 0.001,
            "pnorm": 1.0,
            "output_loss_type": "kl",
            "lr": 0.001,
            "steps": 1000,
            "batch_size": 32,
            "n_eval_steps": 100,
            "print_freq": 100,
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
            "task_config.buffer_size": 2000,
            "batch_size": 64,
        }

        updated_dict = apply_nested_updates(base_config, updates)
        config = Config(**updated_dict)

        assert config.batch_size == 64
        assert isinstance(config.task_config, LMTaskConfig)
        assert config.task_config.max_seq_len == 256
        assert config.task_config.buffer_size == 2000
        assert config.task_config.dataset_name == "test-dataset"  # Unchanged

    def test_config_json_string_loading(self):
        """Test that Config can be loaded from JSON string with nested params."""
        config_dict = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["linear1"],
            "importance_minimality_coeff": 0.001,
            "pnorm": 1.0,
            "output_loss_type": "mse",
            "lr": 0.001,
            "steps": 1000,
            "batch_size": 32,
            "n_eval_steps": 100,
            "print_freq": 100,
            "pretrained_model_class": "spd.experiments.tms.models.TMSModel",
            "task_config": {"task_name": "tms", "feature_probability": 0.1},
        }

        json_string = f"json:{json.dumps(config_dict)}"
        config = load_config(json_string, Config)

        assert isinstance(config, Config)
        assert config.lr == 0.001
        assert isinstance(config.task_config, TMSTaskConfig)
        assert config.task_config.feature_probability == 0.1

    def test_grid_search_end_to_end(self):
        """Test the full grid search flow with nested parameters."""
        # Simulate what the grid search script does
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": {"values": [0.001]},
            "task_config": {"feature_probability": {"values": [0.05, 0.1]}},
        }

        base_config_dict = {
            "C": 10,
            "n_mask_samples": 1,
            "target_module_patterns": ["linear1"],
            "importance_minimality_coeff": 0.001,
            "pnorm": 1.0,
            "output_loss_type": "mse",
            "lr": 0.01,  # Will be overridden
            "steps": 1000,
            "batch_size": 32,
            "n_eval_steps": 100,
            "print_freq": 100,
            "pretrained_model_class": "spd.experiments.tms.models.TMSModel",
            "task_config": {
                "task_name": "tms",
                "feature_probability": 0.2,  # Will be overridden
                "data_generation_type": "at_least_zero_active",
            },
        }

        combinations = generate_grid_combinations(parameters)
        assert len(combinations) == 4

        # Test each combination can create a valid config
        for combo in combinations:
            updated_dict = apply_nested_updates(base_config_dict, combo)
            config = Config(**updated_dict)

            # Check the overrides were applied
            assert config.seed == combo["seed"]
            assert config.lr == combo["lr"]
            assert isinstance(config.task_config, TMSTaskConfig)
            assert (
                config.task_config.feature_probability == combo["task_config.feature_probability"]
            )

            # Check other values are preserved
            assert config.batch_size == 32
            assert config.task_config.data_generation_type == "at_least_zero_active"

            # Verify it can be serialized to JSON and back
            json_str = f"json:{json.dumps(config.model_dump(mode='json'))}"
            reloaded_config = load_config(json_str, Config)
            assert reloaded_config.seed == config.seed
            assert isinstance(reloaded_config.task_config, TMSTaskConfig)
            assert (
                reloaded_config.task_config.feature_probability
                == config.task_config.feature_probability
            )
