"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-autointerp <decomposition_id>
    spd-autointerp <decomposition_id> --config autointerp_config.yaml
"""

import fire


def main(
    decomposition_id: str,
    config: str | None = None,
) -> None:
    """Submit autointerp pipeline (interpret + evals) to SLURM.

    Args:
        decomposition_id: ID of the target decomposition run.
        config: Path to AutointerpSlurmConfig YAML/JSON. Uses built-in defaults if omitted.
    """
    from spd.autointerp.config import AutointerpSlurmConfig
    from spd.autointerp.scripts.run_slurm import submit_autointerp

    slurm_config = (
        AutointerpSlurmConfig.from_file(config) if config is not None else AutointerpSlurmConfig()
    )
    submit_autointerp(decomposition_id=decomposition_id, slurm_config=slurm_config)


def cli() -> None:
    fire.Fire(main)
