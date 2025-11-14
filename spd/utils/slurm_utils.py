"""Shared utilities for SLURM job management."""

from dataclasses import dataclass
import subprocess
import textwrap
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT


@dataclass(frozen=True, slots=True)
class ArrayCommand:
    """Specification for a single SLURM array task command."""

    command: str
    rendezvous_port: int | None = None


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
    commands: list[ArrayCommand],
    snapshot_branch: str,
    nodes: int,
    gpus_per_node: int,
    partition: str,
    time_limit: str = "72:00:00",
    max_concurrent_tasks: int | None = None,
) -> None:
    """Create a SLURM job array script with git snapshot for consistent code.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job array
        commands: List of commands to execute in each array job
        snapshot_branch: Git branch to checkout.
        nodes: Number of nodes to request for each array task.
        gpus_per_node: Number of GPUs per node. Use 0 for CPU-only jobs.
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
    for i, command_spec in enumerate(commands, 1):
        case_statements.append(f"{i}) {command_spec.command} ;;")

    case_block = "\n        ".join(case_statements)

    has_ddp_ports = any(cmd.rendezvous_port for cmd in commands)
    master_port_array = (
        " ".join(str(cmd.rendezvous_port or 0) for cmd in commands) if has_ddp_ports else ""
    )

    gpu_directive = f"#SBATCH --gres=gpu:{gpus_per_node}" if gpus_per_node > 0 else ""
    ntasks_directive = (
        f"#SBATCH --ntasks-per-node={max(1, gpus_per_node)}" if nodes > 0 else ""
    )

    nodes_directive = f"#SBATCH --nodes={nodes}"

    master_port_block = (
        ""
        if not has_ddp_ports
        else textwrap.dedent(
            f"""
            MASTER_PORTS=({master_port_array})
            TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
            if [ $TASK_INDEX -ge 0 ] && [ $TASK_INDEX -lt {len(commands)} ]; then
                SELECTED_PORT=${{MASTER_PORTS[$TASK_INDEX]}}
                if [ "$SELECTED_PORT" -gt 0 ]; then
                    export MASTER_PORT=$SELECTED_PORT
                else
                    unset MASTER_PORT
                fi
            fi
            """
        ).strip()
    )

    script_content = textwrap.dedent(
        f"""
        #!/bin/bash
        {nodes_directive}
        {gpu_directive}
        {ntasks_directive}
        #SBATCH --partition={partition}
        #SBATCH --time={time_limit}
        #SBATCH --job-name={job_name}
        #SBATCH --array={array_range}
        #SBATCH --distribution=pack
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

        # Ensure that dependencies are using the snapshot branch. SLURM might inherit the
        # parent environment, so we need to deactivate and unset the virtual environment.
        deactivate 2>/dev/null || true
        unset VIRTUAL_ENV
        uv sync --no-dev --link-mode copy -q
        source .venv/bin/activate

        if [ {gpus_per_node} -gt 0 ]; then
            export SPD_GPUS_PER_NODE={gpus_per_node}
        fi

        if [ -z "${{MASTER_ADDR:-}}" ]; then
            if [ -n "${{SLURM_NODELIST:-}}" ]; then
                MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
            else
                MASTER_ADDR=$(hostname -s)
            fi
        fi
        export MASTER_ADDR
        {master_port_block}

        # Execute the appropriate command based on array task ID
        case $SLURM_ARRAY_TASK_ID in
        {case_block}
        esac
    """
    ).strip()

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
