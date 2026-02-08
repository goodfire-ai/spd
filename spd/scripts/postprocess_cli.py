"""CLI entry point for unified postprocessing pipeline.

Thin wrapper for fast --help. Heavy imports deferred to postprocess.py.

Usage:
    spd-postprocess <wandb_path>
    spd-postprocess <wandb_path> --no_attributions
    spd-postprocess <wandb_path> --n_harvest_gpus 6 --n_attr_gpus 2
"""

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def main(
    wandb_path: str,
    n_harvest_gpus: int = 4,
    n_attr_gpus: int = 4,
    harvest_n_batches: int | None = None,
    harvest_batch_size: int = 256,
    harvest_ci_threshold: float = 1e-6,
    harvest_time: str = "24:00:00",
    attr_n_batches: int | None = None,
    attr_batch_size: int = 256,
    attr_ci_threshold: float = 0.0,
    attr_time: str = "48:00:00",
    autointerp_model: str = "google/gemini-3-flash-preview",
    autointerp_limit: int | None = None,
    autointerp_time: str = "12:00:00",
    autointerp_no_eval: bool = False,
    no_attributions: bool = False,
    no_autointerp: bool = False,
    partition: str = DEFAULT_PARTITION_NAME,
) -> None:
    """Submit all postprocessing jobs for an SPD run.

    Submits harvest, attributions, and autointerp with SLURM dependency
    chaining. Harvest and attributions run in parallel; autointerp waits
    for harvest to complete.

    Examples:
        spd-postprocess wandb:spd/runs/abc123
        spd-postprocess wandb:spd/runs/abc123 --no_attributions
        spd-postprocess wandb:spd/runs/abc123 --n_harvest_gpus 6 --n_attr_gpus 2
    """
    from spd.scripts.postprocess import postprocess

    postprocess(
        wandb_path=wandb_path,
        n_harvest_gpus=n_harvest_gpus,
        n_attr_gpus=n_attr_gpus,
        harvest_n_batches=harvest_n_batches,
        harvest_batch_size=harvest_batch_size,
        harvest_ci_threshold=harvest_ci_threshold,
        harvest_time=harvest_time,
        attr_n_batches=attr_n_batches,
        attr_batch_size=attr_batch_size,
        attr_ci_threshold=attr_ci_threshold,
        attr_time=attr_time,
        autointerp_model=autointerp_model,
        autointerp_limit=autointerp_limit,
        autointerp_time=autointerp_time,
        autointerp_no_eval=autointerp_no_eval,
        no_attributions=no_attributions,
        no_autointerp=no_autointerp,
        partition=partition,
    )


def cli() -> None:
    fire.Fire(main)
