"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def interpret(
    wandb_path: str,
    model: str = "google/gemini-2.5-flash",
    max_concurrent: int = 20,
    budget_usd: float | None = None,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "12:00:00",
) -> None:
    """Submit interpret job to SLURM (CPU-only).

    Examples:
        spd-interpret wandb:spd/runs/abc123
        spd-interpret wandb:spd/runs/abc123 --budget_usd 100
        spd-interpret wandb:spd/runs/abc123 --max_concurrent 50
    """
    from spd.autointerp.interpret import OpenRouterModelName
    from spd.autointerp.scripts.run_slurm import interpret as interpret_impl

    interpret_impl(
        wandb_path=wandb_path,
        model=OpenRouterModelName(model),
        max_concurrent=max_concurrent,
        budget_usd=budget_usd,
        partition=partition,
        time=time,
    )


def cli() -> None:
    fire.Fire(interpret)
