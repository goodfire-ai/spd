"""Tests for the main() function in spd/scripts/run.py.

This file contains minimal tests focusing on verifying that spd-run correctly
calls either create_slurm_array_script (for SLURM submission) or subprocess.run
(for local execution) with the expected arguments.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from typing import Any
from unittest.mock import Mock, patch

import pytest

from spd.scripts.run import main, make_launch_configs


class TestSPDRun:
    """Test spd-run command execution."""

    _DEFAULT_CLI_KWARGS: dict[str, str | bool] = dict(
        use_wandb=False,
        create_report=False,
    )

    def test_invalid_experiment_name(self):
        """Test that invalid experiment names raise an error.

        I'm keeping this test because it provides valuable validation coverage
        that isn't duplicated elsewhere, and it's a simple unit test that
        doesn't involve complex mocking.
        """
        fake_exp_name = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            main(experiments=fake_exp_name, local=True, **self._DEFAULT_CLI_KWARGS)  # pyright: ignore[reportArgumentType]

        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            main(
                experiments=f"{fake_exp_name},tms_5-2",
                local=True,
                **self._DEFAULT_CLI_KWARGS,  # pyright: ignore[reportArgumentType]
            )

    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_git_snapshot")
    def test_sweep_params_integration(self, mock_create_git_snapshot, mock_subprocess):
        """Test that sweep parameters are correctly integrated into commands.

        This test verifies the integration between sweep parameter loading and
        command generation, which is important functionality not covered by
        the unit tests in other files.
        """
        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_subprocess.return_value = Mock(returncode=0)

        main(
            experiments="tms_5-2",
            # we use the example here bc in CI we don't want to rely on the copy step having run
            sweep="sweep_params.yaml.example",
            local=True,
            **self._DEFAULT_CLI_KWARGS,  # pyright: ignore[reportArgumentType]
        )

        # Verify multiple commands were generated (sweep should create multiple runs)
        assert mock_subprocess.call_count > 1

        # Check that sweep parameters are in the commands
        for call in mock_subprocess.call_args_list:
            cmd_str = call[0][0]
            assert "--sweep_params_json" in cmd_str
            assert "json:" in cmd_str

    def test_make_launch_configs_sweep(self):
        """when given sweep params, make_launch_configs should generate the correct number of
        launch configs. with params swept correctly"""

        sweep_params = {
            "global": {"lr": {"values": [1, 2]}},
            "tms_5-2": {"C": {"values": [10, 20]}, "steps": {"values": [100, 200]}},
        }

        launch_configs = make_launch_configs(
            run_id="test",
            experiments=["tms_5-2"],
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config for j in launch_configs]

        def there_is_one_with(props: dict[str, Any]):
            matching = []
            for config in configs:
                if all(config.__dict__[k] == v for k, v in props.items()):
                    matching.append(config)
            return len(matching) == 1

        assert len(configs) == 8

        assert there_is_one_with({"lr": 1, "C": 10, "steps": 100})
        assert there_is_one_with({"lr": 1, "C": 20, "steps": 100})
        assert there_is_one_with({"lr": 1, "C": 10, "steps": 200})
        assert there_is_one_with({"lr": 1, "C": 20, "steps": 200})
        assert there_is_one_with({"lr": 2, "C": 10, "steps": 100})
        assert there_is_one_with({"lr": 2, "C": 20, "steps": 100})
        assert there_is_one_with({"lr": 2, "C": 10, "steps": 200})
        assert there_is_one_with({"lr": 2, "C": 20, "steps": 200})

    def test_make_launch_configs_sweep_2(self):
        """when given sweep params, make_launch_configs should generate the correct number of
        launch configs. with params swept correctly"""

        sweep_params = {
            "tms_5-2": {"C": {"values": [10]}},
            "tms_40-10": {"steps": {"values": [100, 200]}},
        }

        launch_configs = make_launch_configs(
            run_id="test",
            experiments=["tms_5-2", "tms_40-10"],
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config for j in launch_configs]

        def there_is_one_with(props: dict[str, Any]):
            matching = []
            for config in configs:
                if all(config.__dict__[k] == v for k, v in props.items()):
                    matching.append(config)
            return len(matching) == 1

        assert len(configs) == 3

        assert there_is_one_with({"C": 10})
        assert there_is_one_with({"steps": 100})
        assert there_is_one_with({"steps": 200})
