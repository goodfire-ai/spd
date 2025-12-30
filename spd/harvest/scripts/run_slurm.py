"""SLURM launcher for harvest pipeline.

Submits harvest jobs to SLURM cluster programmatically.

Usage:
    spd-harvest <wandb_path> --n_batches 1000
    spd-harvest <wandb_path> --n_batches 8000 --n_gpus 8
"""

from spd.log import logger
from spd.settings import DEFAULT_PARTITION_NAME
from spd.utils.slurm import SlurmConfig, generate_script, submit_slurm_job


def harvest(
    wandb_path: str,
    n_batches: int,
    n_gpus: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 1000,
    activation_context_tokens_per_side: int = 10,
    pmi_token_top_k: int = 40,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "24:00:00",
) -> None:
    """Submit harvest job to SLURM.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Number of batches to process.
        n_gpus: Number of GPUs for distributed harvesting. If None, uses single GPU.
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
        activation_examples_per_component: Number of activation examples per component.
        activation_context_tokens_per_side: Number of tokens per side of the activation context.
        pmi_token_top_k: Number of top- and bottom-k tokens by PMI to include.
        partition: SLURM partition name.
        time: Job time limit.
    """
    actual_n_gpus = n_gpus if n_gpus else 1
    job_name = "harvest"

    # Build the harvest command with all args
    cmd_parts = [
        "python -m spd.harvest.scripts.run_harvest",
        f'"{wandb_path}"',
        f"--n_batches {n_batches}",
        f"--batch_size {batch_size}",
        f"--ci_threshold {ci_threshold}",
        f"--activation_examples_per_component {activation_examples_per_component}",
        f"--activation_context_tokens_per_side {activation_context_tokens_per_side}",
        f"--pmi_token_top_k {pmi_token_top_k}",
    ]
    if n_gpus:
        cmd_parts.append(f"--n_gpus {n_gpus}")

    harvest_cmd = " \\\n    ".join(cmd_parts)

    # Build full command with echoes
    full_command = "\n".join(
        [
            'echo "=== Harvest ==="',
            f'echo "WANDB_PATH: {wandb_path}"',
            f'echo "N_BATCHES: {n_batches}"',
            f'echo "N_GPUS: {actual_n_gpus}"',
            'echo "SLURM_JOB_ID: $SLURM_JOB_ID"',
            'echo "==============="',
            "",
            harvest_cmd,
            "",
            'echo "Harvest complete!"',
        ]
    )

    config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=actual_n_gpus,
        time=time,
        # Harvest doesn't use git snapshots - runs from REPO_ROOT
    )
    script_content = generate_script(config, full_command)
    result = submit_slurm_job(script_content, "harvest")

    logger.section("Harvest job submitted!")
    logger.values(
        {
            "Job ID": result.job_id,
            "WandB path": wandb_path,
            "N batches": n_batches,
            "N GPUs": actual_n_gpus,
            "Log": result.log_pattern,
            "Script": str(result.script_path),
        }
    )
