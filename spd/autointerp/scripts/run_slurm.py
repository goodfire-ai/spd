"""SLURM launcher for autointerp interpret job.

Submits the interpret job to SLURM. Eval jobs (detection, fuzzing, intruder)
are NOT submitted here — they are orchestrated by the postprocessing pipeline
(spd-postprocess).

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --cost_limit_usd 100
"""

from datetime import datetime

from spd.log import logger
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def submit_interpret(
    wandb_path: str,
    model: str,
    limit: int | None,
    reasoning_effort: str | None,
    config: str | None,
    partition: str,
    time: str,
    cost_limit_usd: float | None,
    dependency_job_id: str | None = None,
    snapshot_branch: str | None = None,
) -> tuple[SubmitResult, str]:
    """Submit autointerp interpret job to SLURM.

    Args:
        dependency_job_id: If provided, the interpret job waits for this job to
            complete before starting (e.g. harvest merge job).
        snapshot_branch: Git snapshot branch to use. If None, no snapshot is set.

    Returns:
        Tuple of (SubmitResult for the interpret job, autointerp_run_id).
    """
    autointerp_run_id = "a-" + datetime.now().strftime("%Y%m%d_%H%M%S")

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
    slurm_config = SlurmConfig(
        job_name="interpret",
        partition=partition,
        n_gpus=0,
        cpus_per_task=16,
        time=time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
    )
    script_content = generate_script(slurm_config, interpret_cmd)
    result = submit_slurm_job(script_content, "interpret")

    logger.section("Interpret job submitted")
    logger.values(
        {
            "Job ID": result.job_id,
            "Autointerp run ID": autointerp_run_id,
            "WandB path": wandb_path,
            "Model": model,
            "Log": result.log_pattern,
        }
    )

    return result, autointerp_run_id
