"""Tests for the main() function in spd/scripts/run.py.

This file contains tests for spd-run, which always submits jobs to SLURM.
For local execution tests, see tests/scripts_simple/.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

from typing import Any
from unittest.mock import patch

import pytest

from spd.scripts.run import _create_training_jobs, _get_experiments


class TestSPDRun:
    """Test spd-run command execution."""

    def test_invalid_experiment_name(self):
        """Test that invalid experiment names raise an error."""
        fake_exp_name = "nonexistent_experiment_please_dont_name_your_experiment_this"
        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            _get_experiments(fake_exp_name)

        with pytest.raises(ValueError, match=f"Invalid experiments.*{fake_exp_name}"):
            _get_experiments(f"{fake_exp_name},tms_5-2")

    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run._wandb_setup")
    def test_sweep_creates_slurm_array(
        self,
        mock_wandb_setup,
        mock_create_git_snapshot,
        mock_create_slurm_array_script,
        mock_submit_slurm_array,
    ):
        """Test that sweep runs create SLURM array jobs with sweep params."""
        from spd.scripts.run_cli import main

        mock_create_git_snapshot.return_value = ("test-branch", "12345678")
        mock_create_slurm_array_script.return_value = "#!/bin/bash\necho test"
        mock_submit_slurm_array.return_value = "12345"

        main(
            experiments="tms_5-2",
            sweep="sweep_params.yaml.example",
            n_agents=2,
        )

        # Verify SLURM array script was created
        mock_create_slurm_array_script.assert_called_once()

        # Verify the run has sweep params and multiple jobs
        call_kwargs = mock_create_slurm_array_script.call_args.kwargs
        training_jobs = call_kwargs["training_jobs"]
        sweep_params = call_kwargs["sweep_params"]
        assert len(training_jobs) > 1  # Sweep should create multiple jobs
        assert sweep_params is not None

    def test_create_training_jobs_sweep(self):
        """Test that sweep params generate correct cartesian product of jobs.

        Verifies:
        - Global params apply to all experiments
        - Experiment-specific params override/extend global
        - Cartesian product generates correct number of jobs
        - C values can be swept per module via module_info
        """
        sweep_params = {
            "global": {"lr": {"values": [1e-3, 1e-4]}},
            "tms_5-2": {
                "module_info": [
                    {"module_pattern": "linear1", "C": {"values": [10, 20]}},
                    {"module_pattern": "linear2", "C": {"values": [10, 20]}},
                ],
            },
        }

        training_jobs = _create_training_jobs(
            experiments=["tms_5-2"],
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config for j in training_jobs]

        def get_c_for_pattern(config: Any, pattern: str) -> int:
            """Get C value for a specific module pattern."""
            for info in config.module_info:
                if info.module_pattern == pattern:
                    return info.C
            raise ValueError(f"Pattern {pattern} not found")

        def there_is_one_with(lr: float, linear1_c: int, linear2_c: int) -> bool:
            matching = [
                c
                for c in configs
                if c.lr == lr
                and get_c_for_pattern(c, "linear1") == linear1_c
                and get_c_for_pattern(c, "linear2") == linear2_c
            ]
            return len(matching) == 1

        # 2 lr values * 2 linear1 C values * 2 linear2 C values = 8 jobs
        assert len(configs) == 8

        assert there_is_one_with(lr=1e-3, linear1_c=10, linear2_c=10)
        assert there_is_one_with(lr=1e-3, linear1_c=10, linear2_c=20)
        assert there_is_one_with(lr=1e-3, linear1_c=20, linear2_c=10)
        assert there_is_one_with(lr=1e-3, linear1_c=20, linear2_c=20)
        assert there_is_one_with(lr=1e-4, linear1_c=10, linear2_c=10)
        assert there_is_one_with(lr=1e-4, linear1_c=10, linear2_c=20)
        assert there_is_one_with(lr=1e-4, linear1_c=20, linear2_c=10)
        assert there_is_one_with(lr=1e-4, linear1_c=20, linear2_c=20)

    def test_create_training_jobs_sweep_multi_experiment(self):
        """Test that sweep params work correctly across multiple experiments.

        Verifies experiment-specific C sweeps via module_info apply to their respective experiments.
        """
        sweep_params = {
            "tms_5-2": {
                "module_info": [
                    {"module_pattern": "linear1", "C": {"values": [10, 20]}},
                ],
            },
            "tms_40-10": {
                "module_info": [
                    {"module_pattern": "linear1", "C": {"values": [100, 200]}},
                ],
            },
        }

        training_jobs = _create_training_jobs(
            experiments=["tms_5-2", "tms_40-10"],
            project="test",
            sweep_params=sweep_params,
        )

        # Separate jobs by experiment
        tms_5_2_jobs = [j for j in training_jobs if "tms_5-2" in j.experiment]
        tms_40_10_jobs = [j for j in training_jobs if "tms_40-10" in j.experiment]

        def get_c_for_pattern(config: Any, pattern: str) -> int:
            """Get C value for a specific module pattern."""
            for info in config.module_info:
                if info.module_pattern == pattern:
                    return info.C
            raise ValueError(f"Pattern {pattern} not found")

        # tms_5-2: 2 jobs (C=10, C=20 for linear1)
        # tms_40-10: 2 jobs (C=100, C=200 for linear1)
        assert len(tms_5_2_jobs) == 2
        assert len(tms_40_10_jobs) == 2
        assert len(training_jobs) == 4

        # Verify tms_5-2 C values
        tms_5_2_cs = {get_c_for_pattern(j.config, "linear1") for j in tms_5_2_jobs}
        assert tms_5_2_cs == {10, 20}

        # Verify tms_40-10 C values
        tms_40_10_cs = {get_c_for_pattern(j.config, "linear1") for j in tms_40_10_jobs}
        assert tms_40_10_cs == {100, 200}
