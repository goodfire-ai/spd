"""Unified postprocessing pipeline for decomposition runs.

Submits all postprocessing steps to SLURM with proper dependency chaining.
All steps always run — data accumulates (harvest upserts, autointerp resumes).

Dependency graph:
    harvest             (GPU array -> merge, GPU, SPD-only)
    ├── intruder eval   (CPU, depends on harvest merge, label-free)
    ├── attributions    (GPU array -> merge, depends on harvest merge, SPD-only)
    └── autointerp      (CPU, LLM calls, resumes via completed keys)
        ├── detection   (CPU, label-dependent)
        └── fuzzing     (CPU, label-dependent)
"""

import secrets
from datetime import datetime
from pathlib import Path

import yaml

from spd.autointerp.scripts.run_slurm import AutointerpSubmitResult, submit_autointerp
from spd.dataset_attributions.scripts.run_slurm import submit_attributions
from spd.harvest.config import SPDHarvestConfig
from spd.harvest.scripts import run_intruder
from spd.harvest.scripts.run_slurm import submit_harvest
from spd.log import logger
from spd.postprocess.config import PostprocessConfig
from spd.settings import SPD_OUT_DIR
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def postprocess(config: PostprocessConfig) -> Path:
    """Submit all postprocessing jobs with SLURM dependency chaining.

    Returns:
        Path to the manifest YAML file.
    """

    snapshot_branch, commit_hash = create_git_snapshot(f"postprocess-{secrets.token_hex(4)}")
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    decomp_cfg = config.harvest.config.method_config

    # === 1. Harvest (always runs, upserts into harvest.db) ===
    harvest_result = submit_harvest(config.harvest, snapshot_branch=snapshot_branch)

    # === 2. Autointerp (depends on harvest, resumes via completed keys) ===
    autointerp_result: AutointerpSubmitResult | None = None
    if config.autointerp is not None:
        autointerp_result = submit_autointerp(
            decomposition_id=decomp_cfg.id,
            config=config.autointerp,
            dependency_job_id=harvest_result.merge_result.job_id,
            snapshot_branch=snapshot_branch,
            harvest_subrun_id=harvest_result.subrun_id,
        )

    # === 3. Intruder eval (depends on harvest merge, label-free) ===
    intruder_result: SubmitResult | None = None
    if config.intruder is not None:
        intruder_cmd = run_intruder.get_command(
            decomposition_id=decomp_cfg.id,
            config=config.intruder.config,
            harvest_subrun_id=harvest_result.subrun_id,
        )

        intruder_slurm = SlurmConfig(
            job_name="spd-intruder-eval",
            partition=config.intruder.partition,
            n_gpus=1,
            time=config.intruder.time,
            snapshot_branch=snapshot_branch,
            dependency_job_id=harvest_result.merge_result.job_id,
        )
        intruder_script = generate_script(intruder_slurm, intruder_cmd)
        intruder_result = submit_slurm_job(intruder_script, "intruder_eval")

        logger.section("Intruder eval job submitted")
        logger.values(
            {
                "Job ID": intruder_result.job_id,
                "Depends on": f"harvest merge ({harvest_result.merge_result.job_id})",
                "Log": intruder_result.log_pattern,
            }
        )

    # === 4. Attributions (depends on harvest merge, SPD-only) ===
    attr_result = None
    if config.attributions is not None:
        assert isinstance(decomp_cfg, SPDHarvestConfig)
        attr_result = submit_attributions(
            wandb_path=decomp_cfg.wandb_path,
            config=config.attributions,
            snapshot_branch=snapshot_branch,
            dependency_job_id=harvest_result.merge_result.job_id,
            harvest_subrun_id=harvest_result.subrun_id,
        )

    # === Write manifest ===
    manifest_id = "pp-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = SPD_OUT_DIR / "postprocess" / manifest_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.yaml"

    jobs: dict[str, str] = {
        "harvest_array": harvest_result.array_result.job_id,
        "harvest_merge": harvest_result.merge_result.job_id,
        "harvest_subrun": harvest_result.subrun_id,
    }
    if intruder_result is not None:
        jobs["intruder_eval"] = intruder_result.job_id
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
        "decomposition": config.harvest.config.method_config.model_dump(),
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
