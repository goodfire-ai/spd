"""Unified postprocessing pipeline for SPD runs.

Submits all postprocessing steps to SLURM with proper dependency chaining.
Creates a single git snapshot shared across all jobs.

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

from spd.log import logger
from spd.postprocess.config import PostprocessConfig
from spd.utils.git_utils import create_git_snapshot


def postprocess(wandb_path: str, config: PostprocessConfig) -> None:
    """Submit all postprocessing jobs with SLURM dependency chaining."""
    from spd.autointerp.scripts.run_slurm import submit_autointerp
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import submit_harvest

    h = config.harvest
    total_gpus = h.n_gpus + (0 if config.attributions is None else config.attributions.n_gpus)
    assert total_gpus <= 8, f"Total GPUs ({total_gpus}) exceeds cluster limit of 8"

    run_id = f"postprocess-{secrets.token_hex(4)}"
    snapshot_branch, commit_hash = create_git_snapshot(run_id)
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest (always) ===
    harvest_result = submit_harvest(wandb_path, h, snapshot_branch=snapshot_branch)

    # === 2. Attributions (parallel with harvest) ===
    if config.attributions is not None:
        submit_attributions(wandb_path, config.attributions, snapshot_branch=snapshot_branch)

    # === 3. Autointerp (depends on harvest merge) ===
    if config.autointerp is not None:
        submit_autointerp(
            wandb_path,
            config.autointerp,
            dependency_job_id=harvest_result.job_id,
            snapshot_branch=snapshot_branch,
        )
