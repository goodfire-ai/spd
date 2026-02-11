"""Unified postprocessing pipeline for SPD runs.

Submits all postprocessing steps to SLURM with proper dependency chaining.
Idempotent: skips steps whose outputs already exist.

Dependency graph:
    harvest (GPU array -> merge)
    └── autointerp (functional unit, depends on harvest merge)
        ├── intruder eval       (CPU, label-free)
        ├── interpret           (CPU, LLM calls)
        │   ├── detection       (CPU, label-dependent)
        │   └── fuzzing         (CPU, label-dependent)
    attributions (GPU array -> merge, parallel with harvest)
"""

import secrets

from spd.autointerp.repo import InterpRepo
from spd.dataset_attributions.repo import AttributionRepo
from spd.harvest.repo import HarvestRepo
from spd.log import logger
from spd.postprocess.config import PostprocessConfig
from spd.utils.git_utils import create_git_snapshot
from spd.utils.wandb_utils import parse_wandb_run_path


def postprocess(wandb_path: str, config: PostprocessConfig) -> None:
    """Submit postprocessing jobs, skipping steps whose outputs already exist."""
    from spd.autointerp.scripts.run_slurm import submit_autointerp
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import submit_harvest

    _, _, run_id = parse_wandb_run_path(wandb_path)

    harvest = HarvestRepo(run_id)
    interp = InterpRepo(run_id)
    attributions = AttributionRepo(run_id)

    harvest_done = harvest.has_activation_contexts() and harvest.has_correlations() and harvest.has_token_stats()
    attributions_done = attributions.has_attributions()
    interp_done = interp.has_interpretations()

    if harvest_done and attributions_done and interp_done:
        logger.info("All postprocessing outputs already exist, nothing to do")
        return

    snapshot_branch, commit_hash = create_git_snapshot(f"postprocess-{secrets.token_hex(4)}")
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest ===
    harvest_job_id: str | None = None
    if harvest_done:
        logger.info("Harvest: already complete, skipping")
    else:
        harvest_result = submit_harvest(wandb_path, config.harvest, snapshot_branch=snapshot_branch)
        harvest_job_id = harvest_result.job_id

    # === 2. Attributions (parallel with harvest) ===
    if config.attributions is not None:
        if attributions_done:
            logger.info("Attributions: already complete, skipping")
        else:
            submit_attributions(wandb_path, config.attributions, snapshot_branch=snapshot_branch)

    # === 3. Autointerp (depends on harvest) ===
    if config.autointerp is not None:
        if interp_done:
            logger.info("Autointerp: already complete, skipping")
        else:
            assert harvest_done or harvest_job_id is not None, (
                "Autointerp requires harvest — but harvest was skipped and no job was submitted"
            )
            submit_autointerp(
                wandb_path,
                config.autointerp,
                dependency_job_id=harvest_job_id,
                snapshot_branch=snapshot_branch,
            )
