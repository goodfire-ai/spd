"""Shared utilities for orchestrating jobs in various compute environments."""

import json
import shlex
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
GPUS_PER_NODE = 8


@dataclass
class Command:
    command: str
    env_vars: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class TrainingJob:
    experiment: str
    script_path: Path
    config: Config


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
    n_gpus: int | None,
    sweep_params: dict[str, Any] | None,
) -> Command:
    """Build the command to run a training job.

    Args:
        run_id: Unique identifier for the run.
        job: The training job to run.
        job_idx: Index of the job in the run.
        n_gpus: Number of GPUs. None or 1 means single GPU/CPU. 2-8 means single-node DDP.
                >8 means multi-node DDP (must be divisible by 8).
        sweep_params: Optional sweep parameters to pass to the job.
    """
    port = _choose_master_port(run_id, job_idx)

    json_tagged_config = f"json:{json.dumps(job.config.model_dump(mode='json'))}"

    if n_gpus is None or n_gpus == 1:
        # Single GPU or CPU
        command = f"python {job.script_path} "
    elif n_gpus <= GPUS_PER_NODE:
        # Single-node DDP
        command = f"torchrun --standalone --nproc_per_node={n_gpus} --master_port={port} {job.script_path} "
    else:
        # Multi-node DDP via srun + torchrun (static launch)
        # SLURM_PROCID is set by srun and corresponds to the task ID (0, 1, ..., n-1)
        # Build the torchrun command with $SLURM vars that will be evaluated on each node
        n_nodes = n_gpus // GPUS_PER_NODE

        # Build torchrun command with shell variables that need to be evaluated on each node
        torchrun_cmd = (
            f"torchrun "
            f"--nnodes={n_nodes} "
            f"--node_rank=$SLURM_PROCID "  # Will be evaluated by bash -c on each node
            f"--nproc_per_node={GPUS_PER_NODE} "
            f'--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) '
            f"--master_port={port} "
            f"{job.script_path} "
            f"--config_json {shlex.quote(json_tagged_config)} "
            f"--sweep_id {run_id} "
            f"--evals_id {job.experiment}"
        )

        if sweep_params is not None:
            json_tagged_sweep_params = f"json:{json.dumps(sweep_params)}"
            torchrun_cmd += f" --sweep_params_json {shlex.quote(json_tagged_sweep_params)}"

        # Two things here:
        # 1: We wrap in srun bash -c with proper shell quoting so that $SLURM_PROCID is evaluated on each
        #    node.
        # 2: We set --cpus-per-task=128 because:
        #    Slurm automatically allocates 16 cpus per gpu. We have `DefCpuPerGPU=16` set
        #    If you run srun bare, it allocates 1 cpu per task. For some reason, we've found that if
        #    we set --cpus-per-task to anything other than 8 * 16 = 128, we get the error:
        #    `srun: error: Unable to create step for job 52790: More processors requested than
        #    permitted`
        command = f"srun --cpus-per-task=128 bash -c {shlex.quote(torchrun_cmd)}"
        return Command(env_vars=CUDA_FLAGS, command=command)

    command += (
        f"--config_json {shlex.quote(json_tagged_config)} "
        f"--sweep_id {run_id} "
        f"--evals_id {job.experiment} "
    )

    if sweep_params is not None:
        json_tagged_sweep_params = f"json:{json.dumps(sweep_params)}"
        command += f" --sweep_params_json {shlex.quote(json_tagged_sweep_params)} "

    return Command(env_vars=CUDA_FLAGS, command=command)


def create_slurm_array_script(
    slurm_job_name: str,
    run_id: str,
    training_jobs: list[TrainingJob],
    sweep_params: dict[str, Any] | None,
    slurm_logs_dir: Path,
    snapshot_branch: str,
    n_gpus: int | None,
    partition: str,
    max_concurrent_tasks: int | None = None,
) -> str:
    """Create a SLURM job array script with git snapshot for consistent code.

    Args:
        slurm_job_name: Name for the SLURM job array
        run_id: Unique identifier for the run.
        training_jobs: List of training jobs to execute.
        sweep_params: Optional sweep parameters to pass to the jobs.
        snapshot_branch: Git branch to checkout.
        n_gpus: Number of GPUs. None or 1 means single GPU. 2-8 means single-node DDP.
                >8 means multi-node DDP (must be divisible by 8).
        partition: SLURM partition to use.
        max_concurrent_tasks: Maximum number of array tasks to run concurrently. If None, no limit.
    """
    n_jobs = len(training_jobs)

    # Create array range (SLURM arrays are 1-indexed)
    if max_concurrent_tasks is not None:
        array_range = f"1-{n_jobs}%{max_concurrent_tasks}"
    else:
        array_range = f"1-{n_jobs}"

    # Create case statement for commands (SLURM is 1-indexed, but we pass 0-indexed to get_command)
    case_block_lines = []
    for i, training_job in enumerate(training_jobs):
        command = get_command(run_id, training_job, i, n_gpus, sweep_params)
        case_block_lines.append(f"{i + 1})")
        if command.env_vars is not None:
            for k, v in command.env_vars.items():
                case_block_lines.append(f"    export {k}={v}")
        case_block_lines.append(f"    {command.command}")
        case_block_lines.append("    ;;")
    case_block = "\n".join(case_block_lines)

    # Compute SLURM resource allocation
    if n_gpus is None or n_gpus == 1:
        n_nodes = 1
        gpus_per_task = 1
    elif n_gpus <= GPUS_PER_NODE:
        n_nodes = 1
        gpus_per_task = n_gpus
    else:
        n_nodes = n_gpus // GPUS_PER_NODE
        gpus_per_task = GPUS_PER_NODE

    script_content = f"""\
#!/bin/bash
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks={n_nodes}
#SBATCH --gres:gpu={gpus_per_task}

#SBATCH --partition={partition}
#SBATCH --time=72:00:00
#SBATCH --job-name={slurm_job_name}
#SBATCH --output={slurm_logs_dir}/slurm-%A_%a.out
#SBATCH --array={array_range}

# Create job-specific working directory on shared filesystem (for multi-node access)
WORK_DIR="$HOME/slurm_workspaces/{slurm_job_name}-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
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
echo "Deactivating virtual environment"
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

# echo "Syncing dependencies"
uv sync --no-dev --link-mode copy -q


echo "Activating virtual environment"
source .venv/bin/activate

echo "Debug: SLURM_NODEID=$SLURM_NODEID"
echo "Debug: SLURM_PROCID=$SLURM_PROCID"
echo "Debug: SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Debug: Master node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

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
