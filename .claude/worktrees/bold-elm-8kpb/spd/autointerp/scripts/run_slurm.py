"""SLURM launcher for autointerp pipeline.

Submits interpret jobs to SLURM cluster programmatically.

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --budget_usd 100
"""

from spd.autointerp.interpret import OpenRouterModelName
from spd.log import logger
from spd.utils.slurm import SlurmConfig, generate_script, submit_slurm_job
from spd.utils.wandb_utils import wandb_path_to_url


def launch_interpret_job(
    wandb_path: str,
    model: OpenRouterModelName,
    partition: str,
    time: str,
    limit: int | None = None,
) -> None:
    """Submit interpret job to SLURM (CPU-only, IO-bound).

    Args:
        wandb_path: WandB run path for the target decomposition run.
        model: OpenRouter model to use for interpretation.
        partition: SLURM partition name.
        time: Job time limit.
        limit: Maximum number of components to interpret (highest mean CI first).
    """
    job_name = "interpret"

    cmd_parts = [
        "python -m spd.autointerp.scripts.run_interpret",
        f'"{wandb_path}"',
        f"--model {model.value}",
    ]
    if limit is not None:
        cmd_parts.append(f"--limit {limit}")
    interpret_cmd = " \\\n    ".join(cmd_parts)

    # Build full command with echoes
    full_command = "\n".join(
        [
            'echo "=== Interpret ==="',
            f'echo "WANDB_PATH: {wandb_path}"',
            f'echo "MODEL: {model.value}"',
            'echo "SLURM_JOB_ID: $SLURM_JOB_ID"',
            'echo "================="',
            "",
            interpret_cmd,
            "",
            'echo "Interpret complete!"',
        ]
    )

    config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=0,  # CPU-only job
        cpus_per_task=16,  # (cluster default is 16cpus/gpu and 15GB memory/cpu. We need the memory)
        time=time,
        snapshot_branch=None,  # Autointerp doesn't use git snapshots
        comment=wandb_path_to_url(wandb_path),
    )
    script_content = generate_script(config, full_command)
    result = submit_slurm_job(script_content, "interpret")

    logger.section("Interpret job submitted!")
    logger.values(
        {
            "Job ID": result.job_id,
            "WandB path": wandb_path,
            "Model": model.value,
            "Log": result.log_pattern,
            "Script": str(result.script_path),
        }
    )
