"""Shared utilities for orchestrating jobs in various compute environments."""

import json
import shlex
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from spd.configs import Config
from spd.utils.slurm import SlurmArrayConfig, generate_array_script, generate_git_snapshot_setup

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
    run_id: str  # Pre-generated unique run identifier (e.g. "s-a1b2c3d4")


def _choose_master_port(run_id_local: str, idx: int) -> int:
    """Choose a unique port per command.

    Uses a stable hash of (run_id, idx) mapped into a high, unprivileged port range so that we can
    run multiple DDP processes on the same machine.
    """
    base: int = 20000
    span: int = 20000  # ports in [20000, 40000)
    h: int = int(sha256(f"{run_id_local}:{idx}".encode()).hexdigest(), 16)
    return base + (h % span)


def _build_script_args(
    run_id: str,
    job: TrainingJob,
    sweep_params: dict[str, Any] | None,
) -> str:
    """Build the common script arguments for training jobs."""
    json_tagged_config = f"json:{json.dumps(job.config.model_dump(mode='json'))}"
    args = (
        f"--config_json {shlex.quote(json_tagged_config)} "
        f"--sweep_id {run_id} "
        f"--evals_id {job.experiment} "
        f"--run_id {job.run_id}"
    )
    if sweep_params is not None:
        json_tagged_sweep_params = f"json:{json.dumps(sweep_params)}"
        args += f" --sweep_params_json {shlex.quote(json_tagged_sweep_params)}"
    return args


def get_command(
    run_id: str,
    job: TrainingJob,
    job_idx: int,
    n_gpus: int | None,
    sweep_params: dict[str, Any] | None,
    snapshot_branch: str,
) -> Command:
    """Build the command to run a training job.

    Args:
        run_id: Unique identifier for the run.
        job: The training job to run.
        job_idx: Index of the job in the run.
        n_gpus: Number of GPUs. None or 1 means single GPU/CPU. 2-8 means single-node DDP.
                >8 means multi-node DDP (must be divisible by 8).
        sweep_params: Optional sweep parameters to pass to the job.
        snapshot_branch: Git branch to checkout (used for multi-node workspace setup).
    """
    port = _choose_master_port(run_id, job_idx)
    script_args = _build_script_args(run_id, job, sweep_params)

    match n_gpus:
        case None | 1:
            command = f"python {job.script_path} {script_args}"

        case n if n <= GPUS_PER_NODE:
            command = (
                f"torchrun --standalone --nproc_per_node={n} --master_port={port} "
                f"{job.script_path} {script_args}"
            )

        case _:
            # Multi-node DDP via srun + torchrun
            # $SLURM_PROCID is the node rank (0, 1, ..., n-1), evaluated on each node by bash -c
            n_nodes = n_gpus // GPUS_PER_NODE
            torchrun_cmd = (
                f"torchrun "
                f"--nnodes={n_nodes} "
                f"--node_rank=$SLURM_PROCID "
                f"--nproc_per_node={GPUS_PER_NODE} "
                f'--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) '
                f"--master_port={port} "
                f"{job.script_path} {script_args}"
            )

            # Each node needs its own /tmp workspace since /tmp is node-local
            work_dir = (
                "/tmp/spd/workspace-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-node$SLURM_PROCID"
            )
            setup = generate_git_snapshot_setup(work_dir, snapshot_branch)
            # Explicit srun flags ensure one task per node across all allocated nodes
            srun_flags = f"--nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1"
            command = f"srun {srun_flags} bash -c {shlex.quote(f'{setup}\n{torchrun_cmd}')}"

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
        cmd = get_command(
            run_id,
            training_job,
            i,
            n_gpus,
            sweep_params,
            snapshot_branch=snapshot_branch,
        )
        commands.append(cmd.command)

    match n_gpus:
        case None | 1:
            n_nodes, gpus_per_node = 1, 1
        case n if n <= GPUS_PER_NODE:
            n_nodes, gpus_per_node = 1, n
        case _:
            n_nodes = n_gpus // GPUS_PER_NODE
            gpus_per_node = GPUS_PER_NODE

    config = SlurmArrayConfig(
        job_name=slurm_job_name,
        partition=partition,
        n_gpus=gpus_per_node,
        n_nodes=n_nodes,
        snapshot_branch=snapshot_branch,
        max_concurrent_tasks=max_concurrent_tasks,
    )

    # CUDA_FLAGS are always set for training jobs
    return generate_array_script(config, commands, env=CUDA_FLAGS)
