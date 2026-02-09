"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Submits only the interpret job. Eval jobs (detection, fuzzing, intruder) are
orchestrated by spd-postprocess.
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
) -> None:
    from spd.autointerp.scripts.run_slurm import submit_interpret

    submit_interpret(
        wandb_path=wandb_path,
        model=model,
        limit=limit,
        reasoning_effort=reasoning_effort,
        config=config,
        partition=partition,
        time=time,
        cost_limit_usd=cost_limit_usd,
    )


def cli() -> None:
    fire.Fire(main)
