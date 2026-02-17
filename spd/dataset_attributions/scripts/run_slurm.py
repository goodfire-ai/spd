"""SLURM launcher for dataset attribution harvesting.

Submits multi-GPU attribution jobs as a SLURM array, with a dependent merge job
that runs after all workers complete. Creates a git snapshot to ensure consistent
code across all workers even if jobs are queued.

Usage:
    spd-attributions <wandb_path> --n_gpus 24
    spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8
"""

import secrets

from spd.log import logger
from spd.settings import DEFAULT_PARTITION_NAME
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)


def submit_attributions(
    wandb_path: str,
    n_gpus: int,
    n_batches: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 0.0,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "48:00:00",
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU attribution harvesting job to SLURM.

    Submits a job array where each task processes a subset of batches, then
    submits a merge job that depends on all workers completing. Creates a git
    snapshot to ensure consistent code across all workers.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Total number of batches to process (divided among workers).
            If None, processes entire training dataset.
        n_gpus: Number of GPUs (each gets its own array task).
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for filtering components.
        partition: SLURM partition name.
        time: Job time limit.
        job_suffix: Optional suffix for SLURM job names (e.g., "1h" -> "spd-attr-1h").
    """
    launch_id = f"attr-{secrets.token_hex(4)}"
    snapshot_branch, commit_hash = create_git_snapshot(launch_id)
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    suffix = f"-{job_suffix}" if job_suffix else ""
    array_job_name = f"spd-attr{suffix}"

    # SLURM arrays are 1-indexed, so task ID 1 -> rank 0, etc.
    worker_commands = []
    for rank in range(n_gpus):
        n_batches_arg = f"--n_batches {n_batches} " if n_batches is not None else ""
        cmd = (
            f"python -m spd.dataset_attributions.scripts.run "
            f'"{wandb_path}" '
            f"{n_batches_arg}"
            f"--batch_size {batch_size} "
            f"--ci_threshold {ci_threshold} "
            f"--rank {rank} "
            f"--world_size {n_gpus}"
        )
        worker_commands.append(cmd)

    array_config = SlurmArrayConfig(
        job_name=array_job_name,
        partition=partition,
        n_gpus=1,  # 1 GPU per worker
        time=time,
        snapshot_branch=snapshot_branch,
    )
    array_script = generate_array_script(array_config, worker_commands)
    array_result = submit_slurm_job(
        array_script,
        "attr_harvest",
        is_array=True,
        n_array_tasks=n_gpus,
    )

    # Submit merge job with dependency on array completion
    merge_cmd = f'python -m spd.dataset_attributions.scripts.run "{wandb_path}" --merge'
    merge_config = SlurmConfig(
        job_name="spd-attr-merge",
        partition=partition,
        n_gpus=0,  # No GPU needed for merge
        time="01:00:00",  # Merge is quick
        snapshot_branch=snapshot_branch,
        dependency_job_id=array_result.job_id,
    )
    merge_script = generate_script(merge_config, merge_cmd)
    merge_result = submit_slurm_job(merge_script, "attr_merge")

    logger.section("Dataset attribution jobs submitted!")
    logger.values(
        {
            "WandB path": wandb_path,
            "N batches": n_batches,
            "N GPUs": n_gpus,
            "Batch size": batch_size,
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Array Job ID": array_result.job_id,
            "Merge Job ID": merge_result.job_id,
            "Worker logs": array_result.log_pattern,
            "Merge log": merge_result.log_pattern,
            "Array script": str(array_result.script_path),
            "Merge script": str(merge_result.script_path),
        }
    )
