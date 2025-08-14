"""Tests for the SPD CLI argument parsing."""

import sys
from unittest.mock import patch

import pytest

from spd.scripts.run import cli


@pytest.mark.parametrize(
    "cli_args,expected_kwargs",
    [
        # SPD_RUN_EXAMPLES tests
        # Example 1: Run subset of experiments locally
        (
            ["--experiments", "tms_5-2,resid_mlp1", "--local"],
            {
                "experiments": "tms_5-2,resid_mlp1",
                "sweep": False,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": True,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 2: Run parameter sweep locally
        (
            ["--experiments", "tms_5-2", "--sweep", "--local"],
            {
                "experiments": "tms_5-2",
                "sweep": True,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": True,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 3: Run subset of experiments (no sweep)
        (
            ["--experiments", "tms_5-2,resid_mlp1"],
            {
                "experiments": "tms_5-2,resid_mlp1",
                "sweep": False,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 4: Run parameter sweep with default sweep_params.yaml
        (
            ["--experiments", "tms_5-2,resid_mlp2", "--sweep"],
            {
                "experiments": "tms_5-2,resid_mlp2",
                "sweep": True,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 5: Run parameter sweep with custom sweep params
        (
            ["--experiments", "tms_5-2", "--sweep", "my_sweep.yaml"],
            {
                "experiments": "tms_5-2",
                "sweep": "my_sweep.yaml",
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 6: Run all experiments (no sweep)
        (
            [],
            {
                "experiments": None,
                "sweep": False,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 7: Use custom W&B project
        (
            ["--experiments", "tms_5-2", "--project", "my-spd-project"],
            {
                "experiments": "tms_5-2",
                "sweep": False,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "my-spd-project",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 8: Run all experiments on CPU
        (
            ["--experiments", "tms_5-2", "--cpu"],
            {
                "experiments": "tms_5-2",
                "sweep": False,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": True,
                "dp": 1,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Example 9: Run with data parallelism over 4 GPUs
        (
            ["--experiments", "ss_mlp", "--dp", "4"],
            {
                "experiments": "ss_mlp",
                "sweep": False,
                "n_agents": None,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 4,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
        # Critical test: Short forms
        (
            ["-e", "tms_5-2", "-n", "5"],
            {
                "experiments": "tms_5-2",
                "sweep": False,
                "n_agents": 5,
                "create_report": True,
                "job_suffix": None,
                "cpu": False,
                "dp": 1,
                "project": "spd",
                "local": False,
                "log_format": "default",
                "create_snapshot": True,
                "use_wandb": True,
                "report_title": None,
            },
        ),
    ],
)
def test_cli_argument_parsing(
    cli_args: list[str], expected_kwargs: dict[str, None | bool | int | str | list[str]]
) -> None:
    """Test that CLI arguments are correctly parsed and passed to main."""
    print(
        "if you are seeing this message, something with the run-spd cli has changed. if you need to change the tests, then be sure to also update `SPD_RUN_EXAMPLES` in `spd/scripts/run.py`"
    )
    # Mock sys.argv to simulate command line arguments
    # Mock the `main` function to capture its arguments
    with (
        patch.object(sys, "argv", ["spd-run"] + cli_args) as _,
        patch("spd.scripts.run.main") as mock_main,
    ):
        # Make mock_main not raise any exceptions
        mock_main.return_value = None

        # Run the CLI
        cli()

        # Check that main was called once with the expected arguments
        mock_main.assert_called_once_with(**expected_kwargs)


def test_help_flag():
    """Test that --help flag works and exits cleanly."""
    with patch.object(sys, "argv", ["spd-run", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            cli()
        # Help should exit with code 0
        assert exc_info.value.code == 0


def test_sweep_params_file_path_handling():
    """Test that sweep parameter file paths are handled correctly."""
    test_cases = [
        (["--sweep"], True),  # No arg -> True
        (["--sweep", "params.yaml"], "params.yaml"),  # With arg -> string
        (["--sweep=custom.yaml"], "custom.yaml"),  # With = syntax -> string
    ]

    for cli_args, expected_sweep_value in test_cases:
        with (
            patch.object(sys, "argv", ["spd-run"] + cli_args) as _,
            patch("spd.scripts.run.main") as mock_main,
        ):
            mock_main.return_value = None
            cli()
            # Check the sweep argument specifically
            called_kwargs = mock_main.call_args[1]
            assert called_kwargs["sweep"] == expected_sweep_value, (
                f"Expected sweep={expected_sweep_value}, got {called_kwargs['sweep']}"
            )
