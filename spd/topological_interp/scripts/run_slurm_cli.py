"""CLI entry point for topological interp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-topological-interp <decomposition_id> --config topological_interp_config.yaml
"""

import fire


def main(decomposition_id: str, config: str) -> None:
    """Submit topological interpretation pipeline to SLURM.

    Args:
        decomposition_id: ID of the target decomposition run.
        config: Path to TopologicalInterpSlurmConfig YAML/JSON.
    """
    from spd.topological_interp.config import TopologicalInterpSlurmConfig
    from spd.topological_interp.scripts.run_slurm import submit_topological_interp

    slurm_config = TopologicalInterpSlurmConfig.from_file(config)
    submit_topological_interp(decomposition_id, slurm_config, dependency_job_ids=[])


def cli() -> None:
    fire.Fire(main)
