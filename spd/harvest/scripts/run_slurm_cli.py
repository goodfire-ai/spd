"""CLI entry point for harvest SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-harvest <wandb_path> --n_gpus 24
    spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8  # Only process 1000 batches
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def harvest(
    wandb_path: str,
    n_gpus: int,
    n_batches: int | None = None,
    batch_size: int = 256,
    ci_threshold: float = 1e-6,
    activation_examples_per_component: int = 1000,
    activation_context_tokens_per_side: int = 10,
    pmi_token_top_k: int = 40,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "24:00:00",
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU harvest job to SLURM.

    Submits a job array where each GPU processes a subset of batches,
    then a merge job that combines results after all workers complete.

    Examples:
        spd-harvest wandb:spd/runs/abc123 --n_gpus 24
        spd-harvest wandb:spd/runs/abc123 --n_batches 1000 --n_gpus 8  # Only process 1000 batches

    Args:
        wandb_path: WandB run path for the target decomposition run.
        n_batches: Total number of batches to process (divided among workers).
            If None, processes entire training dataset.
        n_gpus: Number of GPUs (each gets its own array task).
        batch_size: Batch size for processing.
        ci_threshold: CI threshold for component activation.
        activation_examples_per_component: Number of activation examples per component.
        activation_context_tokens_per_side: Number of tokens per side of the activation context.
        pmi_token_top_k: Number of top- and bottom-k tokens by PMI to include.
        partition: SLURM partition name.
        time: Job time limit for worker jobs.
        job_suffix: Optional suffix for SLURM job names (e.g., "v2" -> "spd-harvest-v2").
    """
    from spd.harvest.scripts.run_slurm import harvest as harvest_impl

    harvest_impl(
        wandb_path=wandb_path,
        n_batches=n_batches,
        n_gpus=n_gpus,
        batch_size=batch_size,
        ci_threshold=ci_threshold,
        activation_examples_per_component=activation_examples_per_component,
        activation_context_tokens_per_side=activation_context_tokens_per_side,
        pmi_token_top_k=pmi_token_top_k,
        partition=partition,
        time=time,
        job_suffix=job_suffix,
    )


def cli() -> None:
    fire.Fire(harvest)
