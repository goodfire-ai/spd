"""Tests for SLURM utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from spd.slurm_utils import (
    create_slurm_script,
    get_slurm_partition,
    submit_experiment_job,
    submit_slurm_job,
)


class TestGetSlurmPartition:
    def test_returns_none_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_slurm_partition() is None

    def test_returns_partition_when_set(self):
        with patch.dict(os.environ, {"SLURM_PARTITION": "gpu"}, clear=True):
            assert get_slurm_partition() == "gpu"


class TestCreateSlurmScript:
    def test_basic_script_creation(self):
        script = create_slurm_script(
            job_name="test_job",
            command="python test.py"
        )
        
        assert "#SBATCH --job-name=test_job" in script
        assert "python test.py" in script
        assert "#!/bin/bash" in script

    def test_script_with_partition(self):
        script = create_slurm_script(
            job_name="test_job",
            command="python test.py",
            partition="gpu"
        )
        
        assert "#SBATCH --partition=gpu" in script

    def test_script_with_gpu(self):
        script = create_slurm_script(
            job_name="test_job",
            command="python test.py",
            gpu=2
        )
        
        assert "#SBATCH --gres=gpu:2" in script

    def test_script_with_output_files(self):
        script = create_slurm_script(
            job_name="test_job",
            command="python test.py",
            output_file="test.out",
            error_file="test.err"
        )
        
        assert "#SBATCH --output=test.out" in script
        assert "#SBATCH --error=test.err" in script

    def test_script_with_additional_options(self):
        script = create_slurm_script(
            job_name="test_job",
            command="python test.py",
            additional_options={"nodes": "2", "ntasks-per-node": "4"}
        )
        
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --ntasks-per-node=4" in script

    def test_command_as_list(self):
        script = create_slurm_script(
            job_name="test_job",
            command=["python", "test.py", "--config", "config.yaml"]
        )
        
        assert "python test.py --config config.yaml" in script

    @patch('spd.slurm_utils.get_slurm_partition')
    def test_uses_env_partition_when_none_specified(self, mock_get_partition):
        mock_get_partition.return_value = "compute"
        
        script = create_slurm_script(
            job_name="test_job",
            command="python test.py"
        )
        
        assert "#SBATCH --partition=compute" in script


class TestSubmitSlurmJob:
    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_successful_submission(self, mock_tempfile, mock_run):
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test_script.sh"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        # Mock subprocess result
        mock_run.return_value.stdout = "Submitted batch job 12345"
        mock_run.return_value.returncode = 0

        job_id = submit_slurm_job(
            job_name="test_job",
            command="python test.py"
        )

        assert job_id == "12345"
        mock_run.assert_called_once_with(
            ["sbatch", "/tmp/test_script.sh"],
            capture_output=True,
            text=True,
            check=True
        )

    def test_dry_run_mode(self):
        job_id = submit_slurm_job(
            job_name="test_job",
            command="python test.py",
            dry_run=True
        )
        
        assert job_id is None

    @patch('subprocess.run')
    def test_handles_subprocess_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("sbatch not found")
        
        job_id = submit_slurm_job(
            job_name="test_job",
            command="python test.py"
        )
        
        assert job_id is None


class TestSubmitExperimentJob:
    def test_unknown_experiment(self):
        job_id = submit_experiment_job(
            experiment_name="unknown",
            config_path="config.yaml"
        )
        
        assert job_id is None

    @patch('spd.slurm_utils.submit_slurm_job')
    def test_tms_experiment(self, mock_submit):
        mock_submit.return_value = "12345"
        
        job_id = submit_experiment_job(
            experiment_name="tms",
            config_path="config.yaml"
        )
        
        mock_submit.assert_called_once()
        args, kwargs = mock_submit.call_args
        
        assert "spd_tms_config" in kwargs["job_name"]
        assert kwargs["command"] == ["python", "spd/experiments/tms/tms_decomposition.py", "config.yaml"]
        assert job_id == "12345"

    @patch('spd.slurm_utils.submit_slurm_job')
    def test_experiment_with_custom_options(self, mock_submit):
        mock_submit.return_value = "12345"
        
        submit_experiment_job(
            experiment_name="tms",
            config_path="config.yaml",
            partition="gpu",
            time="12:00:00",
            memory="32G"
        )
        
        args, kwargs = mock_submit.call_args
        assert kwargs["partition"] == "gpu"
        assert kwargs["time"] == "12:00:00"
        assert kwargs["memory"] == "32G"