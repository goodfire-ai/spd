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

from datetime import datetime

from spd.log import logger
from spd.scripts.postprocess_config import AutointerpEvalConfig
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def _submit_cpu_job(
    job_name: str,
    command: str,
    partition: str,
    time: str,
    snapshot_branch: str | None,
    dependency_job_id: str | None,
) -> SubmitResult:
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=0,
        cpus_per_task=16,
        time=time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
    )
    script_content = generate_script(slurm_config, command)
    return submit_slurm_job(script_content, job_name)


def launch_autointerp_pipeline(
    wandb_path: str,
    model: str,
    limit: int | None,
    reasoning_effort: str | None,
    config: str | None,
    partition: str,
    time: str,
    cost_limit_usd: float | None,
    evals: AutointerpEvalConfig | None,
    dependency_job_id: str | None = None,
    snapshot_branch: str | None = None,
) -> SubmitResult:
    """Submit the autointerp pipeline to SLURM.

    Submits interpret + eval jobs as a functional unit. All jobs depend on a
    prior harvest merge (passed as dependency_job_id).

    Args:
        evals: Eval config. If None, only the interpret job is submitted.
        dependency_job_id: Job to wait for before starting (e.g. harvest merge).
        snapshot_branch: Git snapshot branch to use.

    Returns:
        SubmitResult for the interpret job.
    """
    autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # === 1. Interpret job ===
    interpret_parts = [
        "python -m spd.autointerp.scripts.run_interpret",
        f'"{wandb_path}"',
        f"--autointerp_run_id {autointerp_run_id}",
    ]
    if config is not None:
        interpret_parts.append(f"--config {config}")
    else:
        interpret_parts.append(f"--model {model}")
        if reasoning_effort is not None:
            interpret_parts.append(f"--reasoning_effort {reasoning_effort}")
    if limit is not None:
        interpret_parts.append(f"--limit {limit}")
    if cost_limit_usd is not None:
        interpret_parts.append(f"--cost_limit_usd {cost_limit_usd}")

    interpret_cmd = " \\\n    ".join(interpret_parts)
    interpret_result = _submit_cpu_job(
        "spd-interpret", interpret_cmd, partition, time, snapshot_branch, dependency_job_id
    )

    logger.section("Interpret job submitted")
    logger.values(
        {
            "Job ID": interpret_result.job_id,
            "Autointerp run ID": autointerp_run_id,
            "WandB path": wandb_path,
            "Model": model,
            "Log": interpret_result.log_pattern,
        }
    )

    if evals is None:
        return interpret_result

    # === 2. Detection + fuzzing scoring (depend on interpret) ===
    for scorer in ("detection", "fuzzing"):
        scoring_cmd = " \\\n    ".join(
            [
                "python -m spd.autointerp.scoring.scripts.run_label_scoring",
                f'"{wandb_path}"',
                f"--scorer {scorer}",
                f"--autointerp_run_id {autointerp_run_id}",
                f"--model {evals.eval_model}",
            ]
        )
        scoring_result = _submit_cpu_job(
            f"spd-{scorer}",
            scoring_cmd,
            evals.partition,
            evals.time,
            snapshot_branch,
            dependency_job_id=interpret_result.job_id,
        )
        logger.section(f"{scorer.capitalize()} scoring job submitted")
        logger.values(
            {
                "Job ID": scoring_result.job_id,
                "Depends on": f"interpret ({interpret_result.job_id})",
                "Log": scoring_result.log_pattern,
            }
        )

    return interpret_result
