"""Shared utilities for SLURM job management."""

import subprocess
import textwrap
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT
from spd.utils.command_utils import Command


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


def _create_slurm_script_base(
    script_path: Path,
    job_name: str,
    snapshot_branch: str,
    n_gpus: int,
    partition: str,
    time_limit: str,
    sbatch_directives: str,
    work_dir_suffix: str,
    command_block: str,
) -> None:
    """Create a SLURM script with git snapshot for consistent code.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job
        snapshot_branch: Git branch to checkout
        n_gpus: Number of GPUs. If 0, use CPU jobs.
        partition: SLURM partition to use
        time_limit: Time limit for the job
        sbatch_directives: Additional SBATCH directives (e.g. --array, --dependency, --output)
        work_dir_suffix: Suffix for the working directory (e.g. "${SLURM_JOB_ID}" or "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}")
        command_block: The command(s) to execute
    """
    script_content = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --nodes=1
        #SBATCH --gres=gpu:{n_gpus}
        #SBATCH --partition={partition}
        #SBATCH --time={time_limit}
        #SBATCH --job-name={job_name}
        {sbatch_directives}

        # Create job-specific working directory
        WORK_DIR="/tmp/spd-gf-copy-{work_dir_suffix}"

        # Clone the repository to the job-specific directory
        git clone {REPO_ROOT} $WORK_DIR

        # Change to the cloned repository directory
        cd $WORK_DIR

        # Copy the .env file from the original repository for WandB authentication
        cp {REPO_ROOT}/.env .env

        # Checkout the snapshot branch to ensure consistent code
        git checkout {snapshot_branch}

        # Ensure that dependencies are using the snapshot branch. SLURM might inherit the
        # parent environment, so we need to deactivate and unset the virtual environment.
        deactivate 2>/dev/null || true
        unset VIRTUAL_ENV
        uv sync --no-dev --link-mode copy -q
        source .venv/bin/activate

        {command_block}
    """).strip()

    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)


def create_slurm_array_script(
    script_path: Path,
    job_name: str,
    commands: list[Command],
    snapshot_branch: str,
    n_gpus_per_job: int,
    partition: str,
    time_limit: str = "72:00:00",
    max_concurrent_tasks: int | None = None,
) -> None:
    """Create a SLURM job array script with git snapshot for consistent code.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job array
        commands: List of Command objects to execute in each array job
        snapshot_branch: Git branch to checkout.
        n_gpus_per_job: Number of GPUs per job. If 0, use CPU jobs.
        partition: SLURM partition to use
        time_limit: Time limit for each job (default: 72:00:00)
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
    for i, cmd in enumerate(commands, 1):
        case_statements.append(f"{i}) {cmd.script_line()} ;;")

    case_block = "\n        ".join(case_statements)

    sbatch_directives = f"""#SBATCH --array={array_range}
        #SBATCH --output={slurm_logs_dir}/slurm-%A_%a.out"""

    command_block = f"""# Execute the appropriate command based on array task ID
        case $SLURM_ARRAY_TASK_ID in
        {case_block}
        esac"""

    _create_slurm_script_base(
        script_path=script_path,
        job_name=job_name,
        snapshot_branch=snapshot_branch,
        n_gpus=n_gpus_per_job,
        partition=partition,
        time_limit=time_limit,
        sbatch_directives=sbatch_directives,
        work_dir_suffix="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}",
        command_block=command_block,
    )


def create_slurm_script(
    script_path: Path,
    job_name: str,
    command: "Command",
    snapshot_branch: str,
    n_gpus: int,
    partition: str,
    time_limit: str = "72:00:00",
    dependency_job_id: str | None = None,
) -> None:
    """Create a SLURM job script with git snapshot for consistent code.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job
        command: Command object to execute
        snapshot_branch: Git branch to checkout
        n_gpus: Number of GPUs. If 0, use CPU job.
        partition: SLURM partition to use
        time_limit: Time limit for the job (default: 72:00:00)
        dependency_job_id: Optional job ID to depend on (uses afterok)
    """
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    # Build SBATCH directives
    directives = [f"#SBATCH --output={slurm_logs_dir}/slurm-%j.out"]
    if dependency_job_id is not None:
        directives.append(f"#SBATCH --dependency=afterok:{dependency_job_id}")

    sbatch_directives = "\n        ".join(directives)

    command_block = f"# Execute the command\n        {command.script_line()}"

    _create_slurm_script_base(
        script_path=script_path,
        job_name=job_name,
        snapshot_branch=snapshot_branch,
        n_gpus=n_gpus,
        partition=partition,
        time_limit=time_limit,
        sbatch_directives=sbatch_directives,
        work_dir_suffix="${SLURM_JOB_ID}",
        command_block=command_block,
    )


def submit_slurm_script(script_path: Path) -> str:
    """Submit a SLURM job (array or single) and return the job ID.

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
