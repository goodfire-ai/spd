"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Submits interpret + eval jobs as a functional unit. Use --no_eval to skip evals.
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def main(
    wandb_path: str,
    # model: str = "google/gemini-3-flash-preview",
    eval_model: str = "google/gemini-3-flash-preview",
    limit: int | None = None,
    reasoning_effort: str | None = None,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "12:00:00",
    cost_limit_usd: float | None = None,
    no_eval: bool = False,
) -> None:
    from spd.autointerp.scripts.run_slurm import launch_autointerp_pipeline
    from spd.scripts.postprocess_config import AutointerpEvalConfig

    evals = None if no_eval else AutointerpEvalConfig(eval_model=eval_model, partition=partition)

    launch_autointerp_pipeline(
        wandb_path=wandb_path,
        model=eval_model,
        limit=limit,
        reasoning_effort=reasoning_effort,
        config=None,
        partition=partition,
        time=time,
        cost_limit_usd=cost_limit_usd,
        evals=evals,
    )


def cli() -> None:
    fire.Fire(main)
