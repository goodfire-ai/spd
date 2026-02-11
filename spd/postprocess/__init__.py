"""Unified postprocessing pipeline for SPD runs.

Submits all postprocessing steps to SLURM with proper dependency chaining.
All steps always run — data accumulates (harvest upserts, autointerp resumes).

Dependency graph:
    harvest (GPU array -> merge)
    └── autointerp (functional unit, depends on harvest merge)
        ├── intruder eval       (CPU, label-free)
        ├── interpret           (CPU, LLM calls, resumes via completed keys)
        │   ├── detection       (CPU, label-dependent)
        │   └── fuzzing         (CPU, label-dependent)
    attributions (GPU array -> merge, parallel with harvest)
"""

import secrets

from spd.log import logger
from spd.postprocess.config import PostprocessConfig
from spd.utils.git_utils import create_git_snapshot
from spd.utils.wandb_utils import parse_wandb_run_path


def postprocess(wandb_path: str, config: PostprocessConfig) -> None:
    """Submit all postprocessing jobs with SLURM dependency chaining."""
    from spd.autointerp.scripts.run_slurm import submit_autointerp
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import submit_harvest

    parse_wandb_run_path(wandb_path)

    snapshot_branch, commit_hash = create_git_snapshot(f"postprocess-{secrets.token_hex(4)}")
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest (always runs, upserts into harvest.db) ===
    harvest_result = submit_harvest(wandb_path, config.harvest, snapshot_branch=snapshot_branch)

    # === 2. Attributions (parallel with harvest, overwrites) ===
    if config.attributions is not None:
        submit_attributions(wandb_path, config.attributions, snapshot_branch=snapshot_branch)

    # === 3. Autointerp (depends on harvest, resumes via completed keys) ===
    if config.autointerp is not None:
        submit_autointerp(
            wandb_path,
            config.autointerp,
            dependency_job_id=harvest_result.job_id,
            snapshot_branch=snapshot_branch,
            harvest_subrun_id=harvest_result.subrun_id,
        )
