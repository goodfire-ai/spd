"""SLURM launcher for autointerp pipeline.

Autointerp is a functional unit: interpret + label-dependent evals. This module
submits all jobs in the unit with proper dependency chaining.

Dependency graph (depends on a prior harvest merge):
    interpret         (depends on harvest merge)
    ├── detection     (depends on interpret)
    └── fuzzing       (depends on interpret)

(Intruder eval is label-free and belongs to the harvest functional unit.)

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --cost_limit_usd 100
"""

from dataclasses import dataclass
from datetime import datetime

from spd.autointerp.config import AutointerpSlurmConfig
from spd.log import logger
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


@dataclass
class AutointerpSubmitResult:
    interpret_result: SubmitResult
    detection_result: SubmitResult | None
    fuzzing_result: SubmitResult | None


def submit_autointerp(
    wandb_path: str,
    slurm_config: AutointerpSlurmConfig,
    dependency_job_id: str | None = None,
    snapshot_branch: str | None = None,
    harvest_subrun_id: str | None = None,
) -> AutointerpSubmitResult:
    """Submit the autointerp pipeline to SLURM.

    Submits interpret + eval jobs as a functional unit. All jobs depend on a
    prior harvest merge (passed as dependency_job_id).

    Args:
        wandb_path: WandB run path for the target decomposition run.
        slurm_config: Autointerp SLURM configuration.
        dependency_job_id: Job to wait for before starting (e.g. harvest merge).
        snapshot_branch: Git snapshot branch to use.

    Returns:
        AutointerpSubmitResult with interpret, detection, and fuzzing results.
    """
    interp_config = slurm_config.config
    partition = slurm_config.partition
    time = slurm_config.time
    evals = slurm_config.evals

    autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    config_json = interp_config.model_dump_json(exclude_none=True)

    # === 1. Interpret job ===
    interpret_parts = [
        "python -m spd.autointerp.scripts.run_interpret",
        f'"{wandb_path}"',
        f"--autointerp_run_id {autointerp_run_id}",
        f"--config_json '{config_json}'",
    ]
    if harvest_subrun_id is not None:
        interpret_parts.append(f"--harvest_subrun_id {harvest_subrun_id}")

    interpret_cmd = " \\\n    ".join(interpret_parts)

    interpret_slurm = SlurmConfig(
        job_name="spd-interpret",
        partition=partition,
        n_gpus=0,
        cpus_per_task=16,
        time=time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
    )
    script_content = generate_script(interpret_slurm, interpret_cmd)
    interpret_result = submit_slurm_job(script_content, "spd-interpret")

    logger.section("Interpret job submitted")
    logger.values(
        {
            "Job ID": interpret_result.job_id,
            "Autointerp run ID": autointerp_run_id,
            "WandB path": wandb_path,
            "Model": interp_config.model,
            "Log": interpret_result.log_pattern,
        }
    )

    if evals is None:
        return AutointerpSubmitResult(
            interpret_result=interpret_result,
            detection_result=None,
            fuzzing_result=None,
        )

    # === 2. Detection + fuzzing scoring (depend on interpret) ===
    eval_config_json = evals.model_dump_json(exclude_none=True)

    scoring_results: dict[str, SubmitResult] = {}
    for scorer in ("detection", "fuzzing"):
        scoring_parts = [
            "python -m spd.autointerp.scoring.scripts.run_label_scoring",
            f'"{wandb_path}"',
            f"--scorer {scorer}",
            f"--eval_config_json '{eval_config_json}'",
        ]
        if harvest_subrun_id is not None:
            scoring_parts.append(f"--harvest_subrun_id {harvest_subrun_id}")
        scoring_cmd = " \\\n    ".join(scoring_parts)

        eval_slurm = SlurmConfig(
            job_name=f"spd-{scorer}",
            partition=partition,
            n_gpus=0,
            cpus_per_task=16,
            time=slurm_config.evals_time,
            snapshot_branch=snapshot_branch,
            dependency_job_id=interpret_result.job_id,
        )
        eval_script = generate_script(eval_slurm, scoring_cmd)
        scoring_result = submit_slurm_job(eval_script, f"spd-{scorer}")
        scoring_results[scorer] = scoring_result

        logger.section(f"{scorer.capitalize()} scoring job submitted")
        logger.values(
            {
                "Job ID": scoring_result.job_id,
                "Depends on": f"interpret ({interpret_result.job_id})",
                "Log": scoring_result.log_pattern,
            }
        )

    return AutointerpSubmitResult(
        interpret_result=interpret_result,
        detection_result=scoring_results["detection"],
        fuzzing_result=scoring_results["fuzzing"],
    )
