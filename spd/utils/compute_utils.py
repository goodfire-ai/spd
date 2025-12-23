"""Shared utilities for orchestrating jobs in various compute environments."""

import json
import shlex
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from spd.configs import Config
from spd.utils.slurm import SlurmArrayConfig, generate_array_script

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
    snapshot_branch: str,
    n_gpus: int | None,
    partition: str,
    max_concurrent_tasks: int | None = None,
) -> str:
    """Create a SLURM job array script with git snapshot for consistent code.

    This is a thin wrapper around slurm.generate_array_script that handles
    TrainingJob -> command string conversion and multi-node DDP setup.

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
    # Convert TrainingJobs to command strings
    commands: list[str] = []
    for i, training_job in enumerate(training_jobs):
        cmd = get_command(run_id, training_job, i, n_gpus, sweep_params)
        # For multi-node DDP, get_command returns Command with env_vars
        # We need to prepend env var exports to the command
        if cmd.env_vars:
            env_exports = " ".join(f"{k}={v}" for k, v in cmd.env_vars.items())
            full_cmd = f"export {env_exports} && {cmd.command}"
        else:
            full_cmd = cmd.command
        commands.append(full_cmd)

    # Compute SLURM resource allocation for multi-node DDP
    if n_gpus is None or n_gpus == 1:
        n_nodes = 1
        gpus_per_node = 1
    elif n_gpus <= GPUS_PER_NODE:
        n_nodes = 1
        gpus_per_node = n_gpus
    else:
        n_nodes = n_gpus // GPUS_PER_NODE
        gpus_per_node = GPUS_PER_NODE

    config = SlurmArrayConfig(
        job_name=slurm_job_name,
        partition=partition,
        n_gpus=gpus_per_node,
        n_nodes=n_nodes,
        n_tasks=n_nodes,  # For multi-node DDP, ntasks = n_nodes
        snapshot_branch=snapshot_branch,
        max_concurrent_tasks=max_concurrent_tasks,
    )

    return generate_array_script(config, commands)
