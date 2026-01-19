"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def main(
    wandb_path: str,
    model: str = "google/gemini-3-flash-preview",
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "12:00:00",
    limit: int | None = None,
) -> None:
    from spd.autointerp.interpret import OpenRouterModelName
    from spd.autointerp.scripts.run_slurm import launch_interpret_job

    launch_interpret_job(
        wandb_path=wandb_path,
        model=OpenRouterModelName(model),
        partition=partition,
        time=time,
        limit=limit,
    )


def cli() -> None:
    fire.Fire(main)
