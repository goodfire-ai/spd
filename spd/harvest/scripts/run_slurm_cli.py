"""CLI entry point for harvest SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


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
    """Submit harvest job to SLURM (GPU).

    Examples:
        spd-harvest wandb:spd/runs/abc123 --n_batches 1000
        spd-harvest wandb:spd/runs/abc123 --n_batches 8000 --n_gpus 8
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
    )


def cli() -> None:
    fire.Fire(harvest)
