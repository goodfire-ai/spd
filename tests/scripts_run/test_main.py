"""Tests for the main() function in spd/scripts/run.py.

This file contains minimal tests focusing on verifying that spd-run correctly
calls either create_slurm_array_script (for SLURM submission) or subprocess.run
(for local execution) with the expected arguments.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from spd.scripts.run import main
from spd.utils.git_utils import repo_current_branch


class TestSPDRun:
    """Test spd-run command execution."""

    _DEFAULT_MAIN_KWARGS: dict[str, str | bool] = dict(
        create_snapshot=False,
        use_wandb=False,
        create_report=False,
    )

    @pytest.mark.parametrize(
        "experiments,sweep,n_agents,expected_command_count",
        [
            # Single experiment, no sweep
            ("tms_5-2", False, None, 1),
            # Multiple experiments, no sweep
            ("tms_5-2,resid_mlp1", False, None, 2),
            # Single experiment with sweep (assuming default sweep params with 2-3 combinations)
            ("tms_5-2", True, 4, None),  # Command count depends on sweep params
        ],
    )
    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.load_sweep_params")
    def test_spd_run_not_local_no_sweep(
        self,
        mock_load_sweep_params,
        mock_create_script,
        mock_submit,
        experiments,
        sweep,
        n_agents,
        expected_command_count,
    ):
        """Test that spd-run correctly calls create_slurm_array_script for SLURM submission."""
        # Setup mocks
        mock_submit.return_value = "12345"
        if sweep:
            # Mock sweep params to generate predictable number of commands
            mock_load_sweep_params.return_value = {"C": {"values": [5, 10]}}

        # Call main with standard arguments
        main(
            experiments=experiments,
            sweep=sweep,
            local=False,
            n_agents=n_agents,
            **self._DEFAULT_MAIN_KWARGS,  # pyright: ignore[reportArgumentType]
        )

        # Assert create_slurm_array_script was called
        assert mock_create_script.call_count == 1

        # Verify the arguments passed to create_slurm_array_script
        call_kwargs = mock_create_script.call_args[1]

        # Check script_path is a temporary file
        assert isinstance(call_kwargs["script_path"], Path)
        assert "run_array_" in str(call_kwargs["script_path"])

        # Check job_name
        assert call_kwargs["job_name"] == "spd"

        # Check commands list
        commands = call_kwargs["commands"]
        if expected_command_count is not None:
            assert len(commands) == expected_command_count
        else:
            # For sweep tests, just verify we have multiple commands
            assert len(commands) > 1

        # Verify command structure
        for cmd in commands:
            assert "python" in cmd
            assert "_decomposition.py" in cmd
            assert "json:" in cmd
            assert "--sweep_id" in cmd
            assert "--evals_id" in cmd

        # Check other parameters
        assert call_kwargs["cpu"] is False
        assert call_kwargs["snapshot_branch"] == repo_current_branch()
        assert call_kwargs["max_concurrent_tasks"] == (n_agents or len(experiments.split(",")))

    @pytest.mark.parametrize(
        "experiments,sweep",
        [
            # Single experiment, no sweep
            ("tms_5-2", False),
            # Multiple experiments, no sweep
            ("tms_5-2,resid_mlp1", False),
            # Single experiment with sweep
            ("tms_5-2", True),
        ],
    )
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.load_sweep_params")
    def test_spd_run_local_no_sweep(
        self,
        mock_load_sweep_params,
        mock_subprocess,
        experiments,
        sweep,
    ):
        """Test that spd-run correctly calls subprocess.run for local execution."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)
        if sweep:
            # Mock sweep params to generate predictable number of commands
            mock_load_sweep_params.return_value = {"C": {"values": [5, 10]}}

        # Call main with standard arguments
        main(
            experiments=experiments,
            sweep=sweep,
            local=True,
            **self._DEFAULT_MAIN_KWARGS,  # pyright: ignore[reportArgumentType]
        )

        # Calculate expected number of subprocess calls
        num_experiments = len(experiments.split(","))
        expected_calls = num_experiments * 2 if sweep else num_experiments

        # Assert subprocess.run was called the expected number of times
        assert mock_subprocess.call_count == expected_calls

        # Verify each subprocess call
        for call in mock_subprocess.call_args_list:
            args = call[0][0]  # Get the command list

            # Should be a list of arguments
            assert isinstance(args, list)
            assert args[0] == "python"
            assert "_decomposition.py" in args[1]

            # Check for required arguments in the command
            cmd_str = " ".join(args)
            assert "json:" in cmd_str
            assert "--sweep_id" in cmd_str
            assert "--evals_id" in cmd_str

            if sweep:
                assert "--sweep_params_json" in cmd_str

        # No wandb functions should be called since use_wandb=False

    def test_invalid_experiment_name(self):
        """Test that invalid experiment names raise an error.

        I'm keeping this test because it provides valuable validation coverage
        that isn't duplicated elsewhere, and it's a simple unit test that
        doesn't involve complex mocking.
        """
        fake_exp_name = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            main(experiments=fake_exp_name, local=True, **self._DEFAULT_MAIN_KWARGS)  # pyright: ignore[reportArgumentType]

        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            main(
                experiments=f"{fake_exp_name},tms_5-2",
                local=True,
                **self._DEFAULT_MAIN_KWARGS,  # pyright: ignore[reportArgumentType]
            )

    @patch("spd.scripts.run.subprocess.run")
    def test_sweep_params_integration(self, mock_subprocess):
        """Test that sweep parameters are correctly integrated into commands.

        This test verifies the integration between sweep parameter loading and
        command generation, which is important functionality not covered by
        the unit tests in other files.
        """
        mock_subprocess.return_value = Mock(returncode=0)

        main(
            experiments="tms_5-2",
            # we use the example here bc in CI we don't want to rely on the copy step having run
            sweep="sweep_params.yaml.example",
            local=True,
            **self._DEFAULT_MAIN_KWARGS,  # pyright: ignore[reportArgumentType]
        )

        # Verify multiple commands were generated (sweep should create multiple runs)
        assert mock_subprocess.call_count > 1

        # Check that sweep parameters are in the commands
        for call in mock_subprocess.call_args_list:
            args = call[0][0]
            cmd_str = " ".join(args)
            assert "--sweep_params_json" in cmd_str
            assert "json:" in cmd_str
