"""SLURM launcher for autointerp pipeline.

Submits interpret + eval jobs to SLURM cluster:
- Intruder eval: submitted immediately (label-free, only needs harvest data)
- Interpret: submitted immediately
- Detection + fuzzing scoring: submitted with --dependency on interpret job

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --cost_limit_usd 100
    spd-autointerp <wandb_path> --no_eval  # skip eval jobs
"""

from datetime import datetime

from spd.log import logger
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def _submit_cpu_job(
    job_name: str,
    command: str,
    partition: str,
    time: str,
    dependency_job_id: str | None = None,
) -> SubmitResult:
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=0,
        cpus_per_task=16,
        time=time,
        snapshot_branch=None,
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
    eval_model: str,
    no_eval: bool,
) -> None:
    # Generate autointerp_run_id upfront so scoring jobs can reference it
    autointerp_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    interpret_result = _submit_cpu_job("interpret", interpret_cmd, partition, time)

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

    if no_eval:
        return

    # === 2. Intruder eval (label-free, no dependency on interpret) ===
    intruder_cmd = " \\\n    ".join(
        [
            "python -m spd.autointerp.eval.scripts.run_intruder",
            f'"{wandb_path}"',
            f"--model {eval_model}",
        ]
    )
    intruder_result = _submit_cpu_job("intruder", intruder_cmd, partition, time)

    logger.section("Intruder eval job submitted")
    logger.values({"Job ID": intruder_result.job_id, "Log": intruder_result.log_pattern})

    # === 3. Detection scoring (depends on interpret) ===
    detection_cmd = " \\\n    ".join(
        [
            "python -m spd.autointerp.scoring.scripts.run_label_scoring",
            f'"{wandb_path}"',
            "--scorer detection",
            f"--autointerp_run_id {autointerp_run_id}",
            f"--model {eval_model}",
        ]
    )
    detection_result = _submit_cpu_job(
        "detection",
        detection_cmd,
        partition,
        time,
        dependency_job_id=interpret_result.job_id,
    )

    logger.section("Detection scoring job submitted")
    logger.values(
        {
            "Job ID": detection_result.job_id,
            "Depends on": interpret_result.job_id,
            "Log": detection_result.log_pattern,
        }
    )

    # === 4. Fuzzing scoring (depends on interpret) ===
    fuzzing_cmd = " \\\n    ".join(
        [
            "python -m spd.autointerp.scoring.scripts.run_label_scoring",
            f'"{wandb_path}"',
            "--scorer fuzzing",
            f"--autointerp_run_id {autointerp_run_id}",
            f"--model {eval_model}",
        ]
    )
    fuzzing_result = _submit_cpu_job(
        "fuzzing",
        fuzzing_cmd,
        partition,
        time,
        dependency_job_id=interpret_result.job_id,
    )

    logger.section("Fuzzing scoring job submitted")
    logger.values(
        {
            "Job ID": fuzzing_result.job_id,
            "Depends on": interpret_result.job_id,
            "Log": fuzzing_result.log_pattern,
        }
    )
