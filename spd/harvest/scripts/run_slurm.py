"""SLURM launcher for harvest pipeline.

Harvest is a functional unit: GPU workers -> merge -> intruder eval. This module
submits all jobs in the unit with proper dependency chaining.

Usage:
    spd-harvest <wandb_path> --n_gpus 24
    spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8  # Only process 1000 batches
"""

import secrets
from dataclasses import dataclass
from datetime import datetime

from spd.harvest.config import HarvestSlurmConfig
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


@dataclass
class HarvestSubmitResult:
    merge_result: SubmitResult
    subrun_id: str

    @property
    def job_id(self) -> str:
        return self.merge_result.job_id


def submit_harvest(
    wandb_path: str,
    slurm_config: HarvestSlurmConfig,
    job_suffix: str | None = None,
    snapshot_branch: str | None = None,
) -> HarvestSubmitResult:
    """Submit multi-GPU harvest job to SLURM.

    Submits a job array where each task processes a subset of batches, then
    submits a merge job that depends on all workers completing. Creates a git
    snapshot to ensure consistent code across all workers.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        slurm_config: Harvest SLURM configuration.
        job_suffix: Optional suffix for SLURM job names (e.g., "v2" -> "spd-harvest-v2").
        snapshot_branch: Git snapshot branch to use. If None, creates a new snapshot.

    Returns:
        SubmitResult for the merge job (the terminal job in the harvest pipeline).
    """
    config = slurm_config.config
    n_gpus = slurm_config.n_gpus
    partition = slurm_config.partition
    time = slurm_config.time

    if snapshot_branch is None:
        run_id = f"harvest-{secrets.token_hex(4)}"
        snapshot_branch, commit_hash = create_git_snapshot(run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        commit_hash = "shared"

    subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix = f"-{job_suffix}" if job_suffix else ""
    array_job_name = f"spd-harvest{suffix}"

    # Build worker commands (SLURM arrays are 1-indexed, so task ID 1 -> rank 0, etc.)
    worker_commands = []
    for rank in range(n_gpus):
        cmd = (
            f"python -m spd.harvest.scripts.run "
            f'"{wandb_path}" '
            f"--config_json '{config.model_dump_json(exclude_none=True)}' "
            f"--rank {rank} "
            f"--world_size {n_gpus} "
            f"--subrun_id {subrun_id}"
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
    merge_cmd = f'python -m spd.harvest.scripts.run "{wandb_path}" --merge --subrun_id {subrun_id}'
    merge_config = SlurmConfig(
        job_name="spd-harvest-merge",
        partition=partition,
        n_gpus=0,  # No GPU needed for merge
        time="01:00:00",  # Merge is quick
        mem="128G",  # Merge needs RAM to hold all worker states
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
            f"--harvest_subrun_id {subrun_id}",
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
            "Sub-run ID": subrun_id,
            "N batches": config.n_batches,
            "N GPUs": n_gpus,
            "Batch size": config.batch_size,
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Array Job ID": array_result.job_id,
            "Merge Job ID": merge_result.job_id,
            "Intruder Job ID": intruder_result.job_id,
            "Worker logs": array_result.log_pattern,
            "Merge log": merge_result.log_pattern,
            "Intruder log": intruder_result.log_pattern,
        }
    )

    return HarvestSubmitResult(merge_result=merge_result, subrun_id=subrun_id)
