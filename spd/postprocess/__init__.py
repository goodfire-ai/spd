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
from datetime import datetime
from pathlib import Path

import yaml

from spd.log import logger
from spd.postprocess.config import PostprocessConfig
from spd.settings import SPD_OUT_DIR
from spd.utils.git_utils import create_git_snapshot
from spd.utils.wandb_utils import parse_wandb_run_path


def postprocess(wandb_path: str, config: PostprocessConfig) -> Path:
    """Submit all postprocessing jobs with SLURM dependency chaining.

    Returns:
        Path to the manifest YAML file.
    """
    from spd.autointerp.scripts.run_slurm import AutointerpSubmitResult, submit_autointerp
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import submit_harvest

    _, _, run_id = parse_wandb_run_path(wandb_path)

    snapshot_branch, commit_hash = create_git_snapshot(f"postprocess-{secrets.token_hex(4)}")
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest (always runs, upserts into harvest.db) ===
    harvest_result = submit_harvest(wandb_path, config.harvest, snapshot_branch=snapshot_branch)

    # === 2. Attributions (parallel with harvest, overwrites) ===
    attr_result = None
    if config.attributions is not None:
        attr_result = submit_attributions(
            wandb_path, config.attributions, snapshot_branch=snapshot_branch
        )

    # === 3. Autointerp (depends on harvest, resumes via completed keys) ===
    autointerp_result: AutointerpSubmitResult | None = None
    if config.autointerp is not None:
        autointerp_result = submit_autointerp(
            wandb_path,
            config.autointerp,
            dependency_job_id=harvest_result.job_id,
            snapshot_branch=snapshot_branch,
            harvest_subrun_id=harvest_result.subrun_id,
        )

    # === Write manifest ===
    subrun_id = "pp-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = SPD_OUT_DIR / "postprocess" / run_id / subrun_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.yaml"

    jobs: dict[str, str] = {
        "harvest_array": harvest_result.array_result.job_id,
        "harvest_merge": harvest_result.merge_result.job_id,
        "intruder_eval": harvest_result.intruder_result.job_id,
        "harvest_subrun": harvest_result.subrun_id,
    }
    if attr_result is not None:
        jobs["attr_array"] = attr_result.array_result.job_id
        jobs["attr_merge"] = attr_result.merge_result.job_id
        jobs["attr_subrun"] = attr_result.subrun_id
    if autointerp_result is not None:
        jobs["interpret"] = autointerp_result.interpret_result.job_id
        if autointerp_result.detection_result is not None:
            jobs["detection"] = autointerp_result.detection_result.job_id
        if autointerp_result.fuzzing_result is not None:
            jobs["fuzzing"] = autointerp_result.fuzzing_result.job_id

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "wandb_path": wandb_path,
        "run_id": run_id,
        "snapshot_branch": snapshot_branch,
        "commit_hash": commit_hash,
        "config": config.model_dump(),
        "jobs": jobs,
    }

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    logger.section("Postprocess manifest saved")
    logger.info(str(manifest_path))

    return manifest_path
