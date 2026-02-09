"""SLURM launcher for harvest pipeline.

Harvest is a functional unit: GPU workers → merge → intruder eval. This module
submits all jobs in the unit with proper dependency chaining.

Usage:
    spd-harvest <wandb_path> --n_gpus 24
    spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8  # Only process 1000 batches
"""

import secrets

from spd.log import logger
from spd.settings import DEFAULT_PARTITION_NAME
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    SubmitResult,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)


def harvest(
    wandb_path: str,
    n_gpus: int,
    n_batches: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 1000,
    activation_context_tokens_per_side: int = 10,
    pmi_token_top_k: int = 40,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "24:00:00",
    job_suffix: str | None = None,
    snapshot_branch: str | None = None,
) -> SubmitResult:
    """Submit multi-GPU harvest job to SLURM.

    Submits a job array where each task processes a subset of batches, then
    submits a merge job that depends on all workers completing. Creates a git
    snapshot to ensure consistent code across all workers.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Total number of batches to process (divided among workers).
            If None, processes entire training dataset.
        n_gpus: Number of GPUs (each gets its own array task).
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
        activation_examples_per_component: Number of activation examples per component.
        activation_context_tokens_per_side: Number of tokens per side of the activation context.
        pmi_token_top_k: Number of top- and bottom-k tokens by PMI to include.
        partition: SLURM partition name.
        time: Job time limit for worker jobs.
        job_suffix: Optional suffix for SLURM job names (e.g., "v2" -> "spd-harvest-v2").
        snapshot_branch: Git snapshot branch to use. If None, creates a new snapshot.

    Returns:
        SubmitResult for the merge job (the terminal job in the harvest pipeline).
    """
    if snapshot_branch is None:
        run_id = f"harvest-{secrets.token_hex(4)}"
        snapshot_branch, commit_hash = create_git_snapshot(run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        commit_hash = "shared"

    suffix = f"-{job_suffix}" if job_suffix else ""
    array_job_name = f"spd-harvest{suffix}"

    # Build worker commands (SLURM arrays are 1-indexed, so task ID 1 -> rank 0, etc.)
    worker_commands = []
    for rank in range(n_gpus):
        n_batches_arg = f"--n_batches {n_batches} " if n_batches is not None else ""
        cmd = (
            f"python -m spd.harvest.scripts.run "
            f'"{wandb_path}" '
            f"{n_batches_arg}"
            f"--batch_size {batch_size} "
            f"--ci_threshold {ci_threshold} "
            f"--activation_examples_per_component {activation_examples_per_component} "
            f"--activation_context_tokens_per_side {activation_context_tokens_per_side} "
            f"--pmi_token_top_k {pmi_token_top_k} "
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
        "harvest_worker",
        is_array=True,
        n_array_tasks=n_gpus,
    )

    # Submit merge job with dependency on array completion
    merge_cmd = f'python -m spd.harvest.scripts.run "{wandb_path}" --merge'
    merge_config = SlurmConfig(
        job_name="spd-harvest-merge",
        partition=partition,
        n_gpus=0,  # No GPU needed for merge
        time="01:00:00",  # Merge is quick
        mem="64G",  # Merge needs RAM to hold all worker states
        snapshot_branch=snapshot_branch,
        dependency_job_id=array_result.job_id,
    )
    merge_script = generate_script(merge_config, merge_cmd)
    merge_result = submit_slurm_job(merge_script, "harvest_merge")

    # Submit intruder eval with dependency on merge (label-free, only needs harvest data)
    intruder_cmd = " \\\n    ".join(
        [
            "python -m spd.autointerp.eval.scripts.run_intruder",
            f'"{wandb_path}"',
        ]
    )
    intruder_config = SlurmConfig(
        job_name="spd-intruder-eval",
        partition=partition,
        n_gpus=0,
        cpus_per_task=16,
        time="12:00:00",
        snapshot_branch=snapshot_branch,
        dependency_job_id=merge_result.job_id,
    )
    intruder_script = generate_script(intruder_config, intruder_cmd)
    intruder_result = submit_slurm_job(intruder_script, "intruder_eval")

    logger.section("Harvest jobs submitted!")
    logger.values(
        {
            "WandB path": wandb_path,
            "N batches": n_batches,
            "N GPUs": n_gpus,
            "Batch size": batch_size,
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Array Job ID": array_result.job_id,
            "Merge Job ID": merge_result.job_id,
            "Intruder Job ID": intruder_result.job_id,
            "Worker logs": array_result.log_pattern,
            "Merge log": merge_result.log_pattern,
            "Intruder log": intruder_result.log_pattern,
        }
    )

    return merge_result
