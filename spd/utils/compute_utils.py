"""Shared utilities for orchestrating jobs in various compute environments."""

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from spd.configs import Config
from spd.log import logger
from spd.settings import REPO_ROOT

CUDA_FLAGS = {
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}
N_GPUS_PER_NODE_MULTINODE = 8


def get_config_json(config: Config) -> str:
    return f"json:{json.dumps(config.model_dump(mode='json'))}"


@dataclass
class Command:
    command: str
    env_vars: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class TrainingJob:
    experiment: str
    script_path: Path
    config: Config


class CpuOrSingleGpu: ...


@dataclass(frozen=True, slots=True)
class MultiGpu:
    n_gpus: int


@dataclass(frozen=True, slots=True)
class MultiNode:
    n_nodes: int


ComputeStrategy = CpuOrSingleGpu | MultiGpu | MultiNode


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


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


def get_command(
    run_id: str,
    job: TrainingJob,
    job_idx: int,
    compute_strategy: ComputeStrategy,
    sweep_params: dict[str, Any] | None,
) -> Command:
    port = _choose_master_port(run_id, job_idx)

    match compute_strategy:
        case CpuOrSingleGpu():
            command = f"python {job.script_path} "
        case MultiGpu(n_gpus=n_gpus):
            command = f"torchrun --standalone --nproc_per_node={n_gpus} --master_port={port} {job.script_path} "
        case MultiNode(n_nodes=n_nodes):
            command = (
                f"srun "
                f"torchrun "
                f"--nnodes={n_nodes} "
                f"--nproc_per_node={N_GPUS_PER_NODE_MULTINODE} "
                f"--rdzv_id={run_id}_{job_idx} "
                f"--rdzv_backend=c10d "
                f'--rdzv_endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):{port} '
                f"{job.script_path} "
            )

    command += (
        f"--config_json '{get_config_json(job.config)}' "
        f"--sweep_id {run_id} "
        f"--evals_id {job.experiment} "
    )

    if sweep_params is not None:
        command += f"--sweep_params_json '{json.dumps(sweep_params)}' "

    return Command(env_vars=CUDA_FLAGS, command=command)


def create_slurm_array_script(
    slurm_job_name: str,
    run_id: str,
    training_jobs: list[TrainingJob],
    sweep_params: dict[str, Any] | None,
    snapshot_branch: str,
    compute_strategy: ComputeStrategy,
    partition: str,
    max_concurrent_tasks: int | None = None,
) -> str:
    """Create a SLURM job array script with git snapshot for consistent code.

    Args:
        slurm_job_name: Name for the SLURM job array
        run: Run containing jobs to execute
        snapshot_branch: Git branch to checkout.
        compute_strategy: Compute strategy for the jobs.
        partition: SLURM partition to use.
        max_concurrent_tasks: Maximum number of array tasks to run concurrently. If None, no limit.
    """
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    n_jobs = len(training_jobs)

    # Create array range (SLURM arrays are 1-indexed)
    if max_concurrent_tasks is not None:
        array_range = f"1-{n_jobs}%{max_concurrent_tasks}"
    else:
        array_range = f"1-{n_jobs}"

    # Create case statement for commands (SLURM is 1-indexed, but we pass 0-indexed to get_command)
    case_block_lines = []
    for i, training_job in enumerate(training_jobs):
        command = get_command(run_id, training_job, i, compute_strategy, sweep_params)
        case_block_lines.append(f"{i + 1})\n")
        if command.env_vars is not None:
            for k, v in command.env_vars.items():
                case_block_lines.append(f"    export {k}={v}")
        case_block_lines.append(f"    {command.command}")
        case_block_lines.append("    ;;")
    case_block = "\n".join(case_block_lines)

    match compute_strategy:
        case CpuOrSingleGpu():
            n_nodes = 1
            gpus_per_task = 1
        case MultiGpu(n_gpus=n):
            n_nodes = 1
            gpus_per_task = n
        case MultiNode(n_nodes=n):
            n_nodes = n
            gpus_per_task = 8

    script_content = f"""\
#!/bin/bash
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks={n_nodes}
#SBATCH --gpus-per-task={gpus_per_task}
#SBATCH --cpus-per-task={gpus_per_task * 4}  # 4 CPUs per GPU

#SBATCH --partition={partition}
#SBATCH --time=72:00:00
#SBATCH --job-name={slurm_job_name}
#SBATCH --output={slurm_logs_dir}/slurm-%A_%a.out
#SBATCH --array={array_range}

# Create job-specific working directory
# WORK_DIR="$HOME/slurm_workspaces/{slurm_job_name}-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
# mkdir -p "$WORK_DIR"
# Clean up the workspace when the script exits
# trap 'rm -rf "$WORK_DIR"' EXIT

# Clone the repository to the job-specific directory
# git clone {REPO_ROOT} "$WORK_DIR"

# Change to the cloned repository directory
# cd "$WORK_DIR"

# Copy the .env file from the original repository for WandB authentication
# cp {REPO_ROOT}/.env .env

# Checkout the snapshot branch to ensure consistent code
# git checkout "{snapshot_branch}"

# Ensure that dependencies are using the snapshot branch. SLURM might inherit the
# parent environment, so we need to deactivate and unset the virtual environment.
# echo "Deactivating virtual environment"
# deactivate 2>/dev/null || true
# unset VIRTUAL_ENV

# echo "Syncing dependencies"
# uv sync --no-dev --link-mode copy -q

# WORK_DIR="$HOME/spd/"

echo "Activating virtual environment"
source .venv/bin/activate

echo "Running..."
# Execute the appropriate command based on array task ID
case $SLURM_ARRAY_TASK_ID in
{case_block}
esac
"""

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
