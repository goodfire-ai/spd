"""CLI entry point for unified postprocessing pipeline.

Thin wrapper for fast --help. Heavy imports deferred to postprocess.py.

Usage:
    spd-postprocess config.yaml
    spd-postprocess config.yaml --dependency 311644_1
"""

import fire


def main(config: str, dependency: str | None = None, dry_run: bool = False) -> None:
    """Submit all postprocessing jobs for an SPD run.

    Args:
        config: Path to PostprocessConfig YAML.
        dependency: SLURM job ID to wait for before starting (e.g. a training job).
    """
    import yaml

    from spd.log import logger
    from spd.postprocess import postprocess
    from spd.postprocess.config import PostprocessConfig

    cfg = PostprocessConfig.from_file(config)

    if dry_run:
        logger.info("Dry run: skipping submission\n\nConfig:\n")
        logger.info(yaml.dump(cfg.model_dump(), indent=2, sort_keys=False))
        return

    manifest_path = postprocess(config=cfg, dependency_job_id=dependency)
    logger.info(f"Manifest: {manifest_path}")


def cli() -> None:
    fire.Fire(main)
