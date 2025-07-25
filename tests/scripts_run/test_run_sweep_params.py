"""Tests for sweep parameter loading and merging in spd/scripts/run.py."""

import tempfile
from pathlib import Path

import pytest

from spd.scripts.run import _merge_sweep_params, load_sweep_params


class TestMergeSweepParams:
    """Test the _merge_sweep_params function."""

    def test_merge_simple_values(self):
        """Test merging simple values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        _merge_sweep_params(base, override)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 3, "z": 4}, "c": 5}
        _merge_sweep_params(base, override)
        assert base == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3, "c": 5}

    def test_merge_nested_sweep_values(self):
        """Test merging with nested sweep values."""
        base = {
            "loss": {
                "faithfulness_weight": {"values": [0.1, 0.5]},
                "reconstruction_weight": {"values": [1.0]},
            }
        }
        override = {"loss": {"faithfulness_weight": {"values": [1.0, 2.0]}}}
        _merge_sweep_params(base, override)
        assert base == {
            "loss": {
                "faithfulness_weight": {"values": [1.0, 2.0]},
                "reconstruction_weight": {"values": [1.0]},
            }
        }

    def test_override_dict_with_non_dict(self):
        """Test overriding a dict value with a non-dict value."""
        base = {"a": {"x": 1}}
        override = {"a": "new_value"}
        _merge_sweep_params(base, override)
        assert base == {"a": "new_value"}

    def test_override_non_dict_with_dict(self):
        """Test overriding a non-dict value with a dict."""
        base = {"a": "old_value"}
        override = {"a": {"x": 1}}
        _merge_sweep_params(base, override)
        assert base == {"a": {"x": 1}}


class TestLoadSweepParams:
    """Test the load_sweep_params function."""

    def test_global_params_only(self):
        """Test loading with only global parameters."""
        params_yaml = """
global:
  seed:
    values: [0, 1, 2]
  learning_rate:
    values: [0.001, 0.01]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            result = load_sweep_params("tms_5-2", Path(f.name))
            assert result == {
                "seed": {"values": [0, 1, 2]},
                "learning_rate": {"values": [0.001, 0.01]},
            }

            Path(f.name).unlink()

    def test_experiment_specific_override(self):
        """Test experiment-specific parameters overriding global ones."""
        params_yaml = """
global:
  seed:
    values: [0, 1, 2]
  learning_rate:
    values: [0.001, 0.01]

tms_5-2:
  seed:
    values: [100, 200]
  n_components:
    values: [5, 10]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            result = load_sweep_params("tms_5-2", Path(f.name))
            assert result == {
                "seed": {"values": [100, 200]},  # Overridden
                "learning_rate": {"values": [0.001, 0.01]},  # From global
                "n_components": {"values": [5, 10]},  # Added by experiment
            }

            Path(f.name).unlink()

    def test_nested_parameter_override(self):
        """Test overriding nested parameters."""
        params_yaml = """
global:
  loss:
    faithfulness_weight:
      values: [0.1, 0.5]
    reconstruction_weight:
      values: [1.0, 2.0]

resid_mlp1:
  loss:
    faithfulness_weight:
      values: [1.0, 2.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            result = load_sweep_params("resid_mlp1", Path(f.name))
            assert result == {
                "loss": {
                    "faithfulness_weight": {"values": [1.0, 2.0]},  # Overridden
                    "reconstruction_weight": {"values": [1.0, 2.0]},  # From global
                }
            }

            Path(f.name).unlink()

    def test_no_parameters_error(self):
        """Test error when no parameters are found."""
        params_yaml = """
some_other_key:
  foo: bar
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            with pytest.raises(ValueError, match="No sweep parameters found"):
                load_sweep_params("tms_5-2", Path(f.name))

            Path(f.name).unlink()

    def test_experiment_without_global(self):
        """Test loading experiment-specific params without global params."""
        params_yaml = """
tms_5-2:
  seed:
    values: [100, 200]
  n_components:
    values: [5, 10]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            result = load_sweep_params("tms_5-2", Path(f.name))
            assert result == {
                "seed": {"values": [100, 200]},
                "n_components": {"values": [5, 10]},
            }

            Path(f.name).unlink()

    def test_complex_merge_scenario(self):
        """Test a complex scenario with multiple levels of nesting and overrides."""
        params_yaml = """
global:
  seed:
    values: [0, 1, 2]
  optimizer:
    learning_rate:
      values: [0.001, 0.01]
    momentum:
      values: [0.9]
  loss:
    faithfulness_weight:
      values: [0.1, 0.5]
    reconstruction_weight:
      values: [1.0]
    sparsity:
      lambda:
        values: [0.01]

tms_5-2:
  seed:
    values: [100]
  optimizer:
    learning_rate:
      values: [0.1]
    weight_decay:
      values: [0.0001]
  loss:
    sparsity:
      lambda:
        values: [0.1, 0.2]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            result = load_sweep_params("tms_5-2", Path(f.name))
            expected = {
                "seed": {"values": [100]},
                "optimizer": {
                    "learning_rate": {"values": [0.1]},
                    "momentum": {"values": [0.9]},
                    "weight_decay": {"values": [0.0001]},
                },
                "loss": {
                    "faithfulness_weight": {"values": [0.1, 0.5]},
                    "reconstruction_weight": {"values": [1.0]},
                    "sparsity": {"lambda": {"values": [0.1, 0.2]}},
                },
            }
            assert result == expected

            Path(f.name).unlink()

    def test_global_key_not_treated_as_experiment(self):
        """Test that 'global' key itself is not treated as an experiment name."""
        params_yaml = """
global:
  seed:
    values: [0, 1, 2]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(params_yaml)
            f.flush()

            # Should use global parameters, not look for experiment named "global"
            result = load_sweep_params("global", Path(f.name))
            assert result == {"seed": {"values": [0, 1, 2]}}

            Path(f.name).unlink()
