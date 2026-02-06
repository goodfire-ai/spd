"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def main(
    wandb_path: str,
    model: str,
    limit: int | None,
    reasoning_effort: str | None,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "12:00:00",
    cost_limit_usd: float | None = None,
) -> None:
    from spd.autointerp.interpret import OpenRouterModelName
    from spd.autointerp.scripts.run_slurm import launch_interpret_job

    launch_interpret_job(
        wandb_path=wandb_path,
        model=OpenRouterModelName(model),
        limit=limit,
        reasoning_effort=reasoning_effort,
        partition=partition,
        time=time,
        cost_limit_usd=cost_limit_usd,
    )


def cli() -> None:
    fire.Fire(main)
