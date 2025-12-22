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
    max_examples_per_component: int = 50,
) -> None:
    from spd.autointerp.interpret import OpenRouterModelName
    from spd.autointerp.scripts.run_slurm import launch_interpret_job

    launch_interpret_job(
        wandb_path=wandb_path,
        model=OpenRouterModelName(model),
        partition=partition,
        time=time,
        max_examples_per_component=max_examples_per_component,
    )


def cli() -> None:
    fire.Fire(main)
