"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --config autointerp_config.yaml
"""

import fire


def main(
    wandb_path: str,
    config: str | None = None,
) -> None:
    """Submit autointerp pipeline (interpret + evals) to SLURM.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Path to AutointerpSlurmConfig YAML/JSON. Uses built-in defaults if omitted.
    """
    from spd.autointerp.config import AutointerpSlurmConfig
    from spd.autointerp.scripts.run_slurm import submit_autointerp
    from spd.utils.wandb_utils import parse_wandb_run_path

    parse_wandb_run_path(wandb_path)

    slurm_config = (
        AutointerpSlurmConfig.from_file(config) if config is not None else AutointerpSlurmConfig()
    )
    submit_autointerp(wandb_path=wandb_path, slurm_config=slurm_config)


def cli() -> None:
    fire.Fire(main)
