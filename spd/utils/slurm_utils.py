"""Shared utilities for SLURM job management."""

import subprocess
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT
from spd.utils.command import Command
from spd.utils.distributed_utils import ComputeStrategy


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
    job_name: str,
    commands: list[Command],
    snapshot_branch: str,
    job_strategy: ComputeStrategy,
    partition: str,
    max_concurrent_tasks: int | None = None,
) -> str:
    """Create a SLURM job array script with git snapshot for consistent code.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job array
        commands: List of commands to execute in each array job
        snapshot_branch: Git branch to checkout.
        max_concurrent_tasks: Maximum number of array tasks to run concurrently. If None, no limit.
    """

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
        if len(command.env_vars) > 0:
            # Export environment variables on separate lines before the command
            # so they're available when the shell expands ${VAR} references in the command
            env_exports = "\n        ".join([f"export {k}={v}" for k, v in command.env_vars.items()])
            case_statements.append(f"    {i})\n        {env_exports}\n        {command.command}\n        ;;")
        else:
            case_statements.append(f"    {i})\n        {command.command}\n        ;;")

    case_block = "\n".join(case_statements)

    script_content = f"""\
#!/bin/bash
#SBATCH --nodes={job_strategy.n_nodes()}
#SBATCH --gres=gpu:{job_strategy.n_gpus_per_node()}
#SBATCH --partition={partition}
#SBATCH --time=72:00:00
#SBATCH --job-name={job_name}
#SBATCH --array={array_range}
#SBATCH --output={slurm_logs_dir}/slurm-%A_%a.out

# Create job-specific working directory
WORK_DIR="$HOME/slurm_workspaces/{job_name}-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$WORK_DIR"
# Clean up the workspace when the script exits
trap 'rm -rf "$WORK_DIR"' EXIT

# Clone the repository to the job-specific directory
git clone {REPO_ROOT} "$WORK_DIR"

# Change to the cloned repository directory
cd "$WORK_DIR"

# Copy the .env file from the original repository for WandB authentication
cp {REPO_ROOT}/.env .env

# Checkout the snapshot branch to ensure consistent code
git checkout "{snapshot_branch}"

# Ensure that dependencies are using the snapshot branch. SLURM might inherit the
# parent environment, so we need to deactivate and unset the virtual environment.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --no-dev --link-mode copy -q
source .venv/bin/activate


# Execute the appropriate command based on array task ID
case $SLURM_ARRAY_TASK_ID in
{case_block}
esac
"""
# {statements_block or ""}

    return script_content


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
    logger.section("DEPLOYMENT SUMMARY")

    job_info_dict: dict[str, str] = {}
    for job_info in job_info_list:
        if ":" in job_info:
            experiment, job_id = job_info.split(":", 1)
            job_info_dict[experiment] = job_id
        else:
            job_info_dict["Job ID"] = job_info

    logger.values(
        msg=f"Deployed {len(job_info_list)} jobs:",
        data=job_info_dict,
    )

    logger.info("View logs in: ~/slurm_logs/slurm-<job_id>.out")
