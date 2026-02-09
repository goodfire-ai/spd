"""Unified postprocessing pipeline for SPD runs.

Submits all postprocessing steps (harvest, attributions, autointerp) to SLURM
with proper dependency chaining. Creates a single git snapshot shared across
all GPU jobs.

Dependency graph:
    SPD Run (already completed)
    +-- Harvest (GPU array + merge)
    |   +-- Intruder Eval (CPU, label-free -- submitted by harvest)
    |   +-- Autointerp (CPU, LLM calls)
    |       +-- Detection Scoring (CPU)
    |       +-- Fuzzing Scoring (CPU)
    +-- Dataset Attributions (GPU array + merge)

Harvest and Attributions run in parallel. Autointerp chains off Harvest merge.
"""

import secrets

from spd.log import logger
from spd.scripts.postprocess_config import PostprocessConfig
from spd.utils.git_utils import create_git_snapshot


def postprocess(wandb_path: str, config: PostprocessConfig) -> None:
    """Submit all postprocessing jobs with SLURM dependency chaining."""
    from spd.autointerp.scripts.run_slurm import launch_autointerp_pipeline
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import harvest

    h = config.harvest
    total_gpus = h.n_gpus + (0 if config.attributions is None else config.attributions.n_gpus)
    assert total_gpus <= 8, f"Total GPUs ({total_gpus}) exceeds cluster limit of 8"

    run_id = f"postprocess-{secrets.token_hex(4)}"
    snapshot_branch, commit_hash = create_git_snapshot(run_id)
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest (always) ===
    harvest_result = harvest(
        wandb_path=wandb_path,
        n_gpus=h.n_gpus,
        n_batches=h.n_batches,
        batch_size=h.batch_size,
        ci_threshold=h.ci_threshold,
        activation_examples_per_component=h.activation_examples_per_component,
        activation_context_tokens_per_side=h.activation_context_tokens_per_side,
        pmi_token_top_k=h.pmi_token_top_k,
        partition=h.partition,
        time=h.time,
        snapshot_branch=snapshot_branch,
    )

    # === 2. Attributions (parallel with harvest) ===
    if config.attributions is not None:
        a = config.attributions
        submit_attributions(
            wandb_path=wandb_path,
            n_gpus=a.n_gpus,
            n_batches=a.n_batches,
            batch_size=a.batch_size,
            ci_threshold=a.ci_threshold,
            partition=a.partition,
            time=a.time,
            snapshot_branch=snapshot_branch,
        )

    # === 3. Autointerp (depends on harvest merge) ===
    if config.autointerp is not None:
        ai = config.autointerp
        launch_autointerp_pipeline(
            wandb_path=wandb_path,
            model=ai.model,
            limit=ai.limit,
            reasoning_effort=ai.reasoning_effort,
            config=None,
            partition=ai.partition,
            time=ai.time,
            cost_limit_usd=ai.cost_limit_usd,
            eval_model=ai.eval_model,
            no_eval=ai.no_eval,
            dependency_job_id=harvest_result.job_id,
        )
