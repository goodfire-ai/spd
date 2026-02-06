"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def main(
    wandb_path: str,
    model: str = "google/gemini-3-flash-preview",
    limit: int | None = None,
    reasoning_effort: str | None = None,
    config: str | None = None,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "12:00:00",
    cost_limit_usd: float | None = None,
    eval_model: str = "google/gemini-3-flash-preview",
    no_eval: bool = False,
) -> None:
    from spd.autointerp.scripts.run_slurm import launch_autointerp_pipeline

    launch_autointerp_pipeline(
        wandb_path=wandb_path,
        model=model,
        limit=limit,
        reasoning_effort=reasoning_effort,
        config=config,
        partition=partition,
        time=time,
        cost_limit_usd=cost_limit_usd,
        eval_model=eval_model,
        no_eval=no_eval,
    )


def cli() -> None:
    fire.Fire(main)
