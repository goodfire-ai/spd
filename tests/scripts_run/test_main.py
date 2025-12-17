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
        """
        sweep_params = {
            "global": {"lr": {"values": [1e-3, 1e-4]}},
            "tms_5-2": {"seed": {"values": [0, 42]}, "steps": {"values": [100, 200]}},
        }

        training_jobs = _create_training_jobs(
            experiments=["tms_5-2"],
            project="test",
            sweep_params=sweep_params,
        )

        configs = [j.config for j in training_jobs]

        def there_is_one_with(props: dict[str, Any]):
            matching = []
            for config in configs:
                matches = True
                for k, v in props.items():
                    if config.__dict__[k] != v:
                        matches = False
                        break
                if matches:
                    matching.append(config)
            return len(matching) == 1

        # 2 lr values * 2 seed values * 2 steps values = 8 jobs
        assert len(configs) == 8

        assert there_is_one_with({"lr": 1e-3, "seed": 0, "steps": 100})
        assert there_is_one_with({"lr": 1e-3, "seed": 42, "steps": 100})
        assert there_is_one_with({"lr": 1e-3, "seed": 0, "steps": 200})
        assert there_is_one_with({"lr": 1e-3, "seed": 42, "steps": 200})
        assert there_is_one_with({"lr": 1e-4, "seed": 0, "steps": 100})
        assert there_is_one_with({"lr": 1e-4, "seed": 42, "steps": 100})
        assert there_is_one_with({"lr": 1e-4, "seed": 0, "steps": 200})
        assert there_is_one_with({"lr": 1e-4, "seed": 42, "steps": 200})

    def test_create_training_jobs_sweep_multi_experiment(self):
        """Test that sweep params work correctly across multiple experiments.

        Verifies experiment-specific sweep params only apply to their respective experiments.
        """
        sweep_params = {
            "tms_5-2": {"seed": {"values": [0, 42]}},
            "tms_40-10": {"steps": {"values": [100, 200]}},
        }

        training_jobs = _create_training_jobs(
            experiments=["tms_5-2", "tms_40-10"],
            project="test",
            sweep_params=sweep_params,
        )

        # Separate jobs by experiment
        tms_5_2_jobs = [j for j in training_jobs if "tms_5-2" in j.experiment]
        tms_40_10_jobs = [j for j in training_jobs if "tms_40-10" in j.experiment]

        # tms_5-2: 2 jobs (seed=0, seed=42)
        # tms_40-10: 2 jobs (steps=100, steps=200)
        assert len(tms_5_2_jobs) == 2
        assert len(tms_40_10_jobs) == 2
        assert len(training_jobs) == 4

        # Verify tms_5-2 seeds
        tms_5_2_seeds = {j.config.seed for j in tms_5_2_jobs}
        assert tms_5_2_seeds == {0, 42}

        # Verify tms_40-10 steps
        tms_40_10_steps = {j.config.steps for j in tms_40_10_jobs}
        assert tms_40_10_steps == {100, 200}
