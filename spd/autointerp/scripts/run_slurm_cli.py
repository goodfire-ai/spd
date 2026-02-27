"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --config autointerp_config.yaml
"""

import fire


def main(decomposition_id: str, config: str) -> None:
    """Submit autointerp pipeline (interpret + evals) to SLURM.

    Args:
        decomposition_id: ID of the target decomposition run.
        config: Path to AutointerpSlurmConfig YAML/JSON.
    """
    from spd.autointerp.config import AutointerpSlurmConfig
    from spd.autointerp.scripts.run_slurm import submit_autointerp

    slurm_config = AutointerpSlurmConfig.from_file(config)
    submit_autointerp(decomposition_id, slurm_config)


def cli() -> None:
    fire.Fire(main)
