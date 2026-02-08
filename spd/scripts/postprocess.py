"""Unified postprocessing pipeline for SPD runs.

Submits all postprocessing steps (harvest, attributions, autointerp) to SLURM
with proper dependency chaining. Creates a single git snapshot shared across
all GPU jobs.

Dependency graph:
    SPD Run (already completed)
    ├── Harvest (GPU array + merge)
    │   ├── Intruder Eval (CPU, label-free — submitted by harvest)
    │   └── Autointerp (CPU, LLM calls)
    │       ├── Detection Scoring (CPU)
    │       └── Fuzzing Scoring (CPU)
    └── Dataset Attributions (GPU array + merge)

Harvest and Attributions run in parallel. Autointerp chains off Harvest merge.
"""

import secrets

from spd.log import logger
from spd.utils.git_utils import create_git_snapshot


def postprocess(
    wandb_path: str,
    n_harvest_gpus: int,
    n_attr_gpus: int,
    harvest_n_batches: int | None,
    harvest_batch_size: int,
    harvest_ci_threshold: float,
    harvest_time: str,
    attr_n_batches: int | None,
    attr_batch_size: int,
    attr_ci_threshold: float,
    attr_time: str,
    autointerp_model: str,
    autointerp_limit: int | None,
    autointerp_time: str,
    autointerp_no_eval: bool,
    no_attributions: bool,
    no_autointerp: bool,
    partition: str,
) -> None:
    """Submit all postprocessing jobs with SLURM dependency chaining.

    Creates one git snapshot shared by all GPU jobs, then submits:
    1. Harvest (always) — GPU array + merge + intruder eval
    2. Attributions (parallel with harvest, skippable) — GPU array + merge
    3. Autointerp (depends on harvest merge, skippable) — interpret + scoring
    """
    from spd.autointerp.scripts.run_slurm import launch_autointerp_pipeline
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import harvest

    total_gpus = n_harvest_gpus + (0 if no_attributions else n_attr_gpus)
    assert total_gpus <= 8, f"Total GPUs ({total_gpus}) exceeds cluster limit of 8"

    # One snapshot for all GPU jobs
    run_id = f"postprocess-{secrets.token_hex(4)}"
    snapshot_branch, commit_hash = create_git_snapshot(run_id)
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest (always) ===
    harvest_result = harvest(
        wandb_path=wandb_path,
        n_gpus=n_harvest_gpus,
        n_batches=harvest_n_batches,
        batch_size=harvest_batch_size,
        ci_threshold=harvest_ci_threshold,
        partition=partition,
        time=harvest_time,
        snapshot_branch=snapshot_branch,
    )

    # === 2. Attributions (parallel with harvest) ===
    if not no_attributions:
        submit_attributions(
            wandb_path=wandb_path,
            n_gpus=n_attr_gpus,
            n_batches=attr_n_batches,
            batch_size=attr_batch_size,
            ci_threshold=attr_ci_threshold,
            partition=partition,
            time=attr_time,
            snapshot_branch=snapshot_branch,
        )

    # === 3. Autointerp (depends on harvest merge) ===
    if not no_autointerp:
        launch_autointerp_pipeline(
            wandb_path=wandb_path,
            model=autointerp_model,
            limit=autointerp_limit,
            reasoning_effort=None,
            config=None,
            partition=partition,
            time=autointerp_time,
            cost_limit_usd=None,
            eval_model=autointerp_model,
            no_eval=autointerp_no_eval,
            dependency_job_id=harvest_result.job_id,
        )
