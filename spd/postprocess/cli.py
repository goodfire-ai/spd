"""CLI entry point for unified postprocessing pipeline.

Thin wrapper for fast --help. Heavy imports deferred to postprocess.py.

Usage:
    spd-postprocess <wandb_path>
    spd-postprocess <wandb_path> --config my_config.yaml
"""

import fire


def main(
    wandb_path: str,
    config: str | None = None,
    dry_run: bool = False,
) -> None:
    """Submit all postprocessing jobs for an SPD run.

    Args:
        wandb_path: WandB run path (e.g. wandb:goodfire/spd/runs/abc123).
        config: Path to PostprocessConfig YAML. Uses built-in defaults if omitted.

    Examples:
        spd-postprocess wandb:goodfire/spd/runs/abc123
        spd-postprocess wandb:goodfire/spd/runs/abc123 --config my_config.yaml
    """
    import yaml

    from spd.log import logger
    from spd.postprocess import postprocess
    from spd.postprocess.config import PostprocessConfig
    from spd.utils.wandb_utils import parse_wandb_run_path

    parse_wandb_run_path(wandb_path)

    cfg = PostprocessConfig.from_file(config) if config is not None else PostprocessConfig()

    if dry_run:
        logger.info("Dry run: skipping submission\n\nConfig:\n")
        logger.info(yaml.dump(cfg.model_dump(), indent=2, sort_keys=False))
        return

    manifest_path = postprocess(wandb_path=wandb_path, config=cfg)
    logger.info(f"Manifest: {manifest_path}")


def cli() -> None:
    fire.Fire(main)
