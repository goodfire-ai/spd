"""SLURM launcher for dataset attribution harvesting.

Submits multi-GPU attribution jobs as a SLURM array, with a dependent merge job
that runs after all workers complete. Creates a git snapshot to ensure consistent
code across all workers even if jobs are queued.

Usage:
    spd-attributions <wandb_path> --n_gpus 24
    spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8
"""

import secrets

from spd.dataset_attributions.config import AttributionsSlurmConfig
from spd.log import logger
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    SubmitResult,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)


def submit_attributions(
    wandb_path: str,
    slurm_config: AttributionsSlurmConfig,
    job_suffix: str | None = None,
    snapshot_branch: str | None = None,
) -> SubmitResult:
    """Submit multi-GPU attribution harvesting job to SLURM.

    Submits a job array where each task processes a subset of batches, then
    submits a merge job that depends on all workers completing. Creates a git
    snapshot to ensure consistent code across all workers.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        slurm_config: Attribution SLURM configuration.
        job_suffix: Optional suffix for SLURM job names (e.g., "1h" -> "spd-attr-1h").
        snapshot_branch: Git snapshot branch to use. If None, creates a new snapshot.

    Returns:
        SubmitResult for the merge job (the terminal job in the attribution pipeline).
    """
    config = slurm_config.config
    n_gpus = slurm_config.n_gpus
    partition = slurm_config.partition
    time = slurm_config.time

    if snapshot_branch is None:
        run_id = f"attr-{secrets.token_hex(4)}"
        snapshot_branch, commit_hash = create_git_snapshot(run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        commit_hash = "shared"

    suffix = f"-{job_suffix}" if job_suffix else ""
    array_job_name = f"spd-attr{suffix}"

    config_json = config.model_dump_json()

    # SLURM arrays are 1-indexed, so task ID 1 -> rank 0, etc.
    worker_commands = []
    for rank in range(n_gpus):
        cmd = (
            f"python -m spd.dataset_attributions.scripts.run "
            f'"{wandb_path}" '
            f"--config_json '{config_json}' "
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
            "N batches": config.n_batches,
            "N GPUs": n_gpus,
            "Batch size": config.batch_size,
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Array Job ID": array_result.job_id,
            "Merge Job ID": merge_result.job_id,
            "Worker logs": array_result.log_pattern,
            "Merge log": merge_result.log_pattern,
            "Array script": str(array_result.script_path),
            "Merge script": str(merge_result.script_path),
        }
    )

    return merge_result
