"""CLI entry point for dataset attribution SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-attributions <wandb_path> --n_gpus 24
    spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8  # Only process 1000 batches
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def submit_attributions(
    wandb_path: str,
    n_gpus: int,
    n_batches: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 0.0,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "48:00:00",
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU dataset attribution harvesting to SLURM.

    Submits a job array where each GPU processes a subset of batches,
    then a merge job that combines results after all workers complete.

    Examples:
        spd-attributions wandb:spd/runs/abc123 --n_gpus 24
        spd-attributions wandb:spd/runs/abc123 --n_batches 1000 --n_gpus 8  # Only process 1000 batches

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Total number of batches to process (divided among workers).
            If None, processes entire training dataset.
        n_gpus: Number of GPUs (each gets its own array task).
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for filtering components.
        partition: SLURM partition name.
        time: Job time limit for worker jobs.
        job_suffix: Optional suffix for SLURM job names (e.g., "v2" -> "spd-attr-v2").
    """
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions as impl

    impl(
        wandb_path=wandb_path,
        n_batches=n_batches,
        n_gpus=n_gpus,
        batch_size=batch_size,
        ci_threshold=ci_threshold,
        partition=partition,
        time=time,
        job_suffix=job_suffix,
    )


def cli() -> None:
    fire.Fire(submit_attributions)
