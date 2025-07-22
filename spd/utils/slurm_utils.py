"""Shared utilities for SLURM job management."""

import subprocess
import textwrap
from pathlib import Path

from spd.settings import REPO_ROOT
from spd.utils.git_utils import create_git_snapshot


def format_runtime_str(runtime_minutes: int) -> str:
    """Format runtime in minutes to a human-readable string like '2h30m' or '45m'.

    Args:
        runtime_minutes: Runtime in minutes

    Returns:
        Formatted string like '2h30m' for 150 minutes or '45m' for 45 minutes
    """
    minutes = runtime_minutes % 60
    hours = runtime_minutes // 60
    return f"{hours}h{minutes}m" if hours > 0 else f"{minutes}m"


def create_slurm_array_script(
    script_path: Path,
    job_name: str,
    commands: list[str],
    cpu: bool = False,
    time_limit: str = "24:00:00",
    snapshot_branch: str | None = None,
    max_concurrent_tasks: int | None = None,
) -> None:
    """Create a SLURM job array script with git snapshot for consistent code.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job array
        commands: List of commands to execute in each array job
        cpu: If True, use CPU only, otherwise use GPU
        time_limit: Time limit for each job (default: 24:00:00)
        snapshot_branch: Git branch to checkout. If None, creates a new snapshot.
        max_concurrent_tasks: Maximum number of array tasks to run concurrently. If None, no limit.
    """
    if snapshot_branch is None:
        snapshot_branch, _ = create_git_snapshot(branch_name_prefix="snapshot")

    gpu_config = "#SBATCH --gres=gpu:0" if cpu else "#SBATCH --gres=gpu:1"
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    # Create array range (SLURM arrays are 1-indexed)
    if max_concurrent_tasks is not None:
        array_range = f"1-{len(commands)}%{max_concurrent_tasks}"
    else:
        array_range = f"1-{len(commands)}"

    # Create case statement for commands
    case_statements = []
    for i, command in enumerate(commands, 1):
        case_statements.append(f"{i}) {command} ;;")

    case_block = "\n        ".join(case_statements)

    script_content = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --nodes=1
        {gpu_config}
        #SBATCH --time={time_limit}
        #SBATCH --job-name={job_name}
        #SBATCH --array={array_range}
        #SBATCH --output={slurm_logs_dir}/slurm-%A_%a.out

        # Create job-specific working directory
        WORK_DIR="/tmp/spd-gf-copy-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"

        # Clone the repository to the job-specific directory
        git clone {REPO_ROOT} $WORK_DIR

        # Change to the cloned repository directory
        cd $WORK_DIR

        # Copy the .env file from the original repository for WandB authentication
        cp {REPO_ROOT}/.env .env

        # Checkout the snapshot branch to ensure consistent code
        git checkout {snapshot_branch}

        # Execute the appropriate command based on array task ID
        case $SLURM_ARRAY_TASK_ID in
        {case_block}
        esac
    """).strip()

    with open(script_path, "w") as f:
        f.write(script_content)

    # Make script executable
    script_path.chmod(0o755)


def submit_slurm_array(script_path: Path) -> str:
    """Submit a SLURM job array and return the array job ID.

    Args:
        script_path: Path to SLURM batch script

    Returns:
        Array job ID from submitted job array
    """
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=True
    )
    # Extract job ID from sbatch output (format: "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def create_analysis_slurm_script(
    script_path: Path,
    job_name: str,
    command: str,
    dependency_job_id: str,
    cpu: bool = True,
    time_limit: str = "01:00:00",
    snapshot_branch: str | None = None,
) -> None:
    """Create a SLURM script for analysis job with dependency.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job
        command: Command to execute in the job
        dependency_job_id: Job ID to depend on (will wait for this job to complete)
        cpu: If True, use CPU only, otherwise use GPU
        time_limit: Time limit for the job (default: 01:00:00)
        snapshot_branch: Git branch to checkout. If None, creates a new snapshot.
    """
    if snapshot_branch is None:
        snapshot_branch, _ = create_git_snapshot(branch_name_prefix="analysis")

    gpu_config = "#SBATCH --gres=gpu:0" if cpu else "#SBATCH --gres=gpu:1"
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    script_content = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --nodes=1
        {gpu_config}
        #SBATCH --time={time_limit}
        #SBATCH --job-name={job_name}
        #SBATCH --dependency=afterok:{dependency_job_id}
        #SBATCH --output={slurm_logs_dir}/slurm-%j.out

        # Create job-specific working directory
        WORK_DIR="/tmp/spd-analysis-${{SLURM_JOB_ID}}"

        # Clone the repository to the job-specific directory
        git clone {REPO_ROOT} $WORK_DIR

        # Change to the cloned repository directory
        cd $WORK_DIR

        # Copy the .env file from the original repository for WandB authentication
        cp {REPO_ROOT}/.env .env

        # Checkout the snapshot branch to ensure consistent code
        git checkout {snapshot_branch}

        # Execute the analysis command
        {command}
    """).strip()

    with open(script_path, "w") as f:
        f.write(script_content)

    # Make script executable
    script_path.chmod(0o755)


def submit_slurm_job(script_path: Path) -> str:
    """Submit a SLURM job and return the job ID.

    Args:
        script_path: Path to SLURM batch script

    Returns:
        Job ID from submitted job
    """
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=True
    )
    # Extract job ID from sbatch output (format: "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def print_job_summary(job_info_list: list[str]) -> None:
    """Print summary of submitted jobs.

    Args:
        job_info_list: List of job information strings (can be just job IDs
                      or formatted as "experiment:job_id")
    """
    print("=" * 50)
    print("DEPLOYMENT SUMMARY")
    print("=" * 50)
    print(f"Deployed {len(job_info_list)} jobs:")

    for job_info in job_info_list:
        if ":" in job_info:
            experiment, job_id = job_info.split(":", 1)
            print(f"  {experiment}: {job_id}")
        else:
            print(f"  Job ID: {job_info}")

    print("\nView logs in: ~/slurm_logs/slurm-<job_id>.out")
