"""CLI entry point for unified postprocessing pipeline.

Thin wrapper for fast --help. Heavy imports deferred to postprocess.py.

Usage:
    spd-postprocess <wandb_path>
    spd-postprocess <wandb_path> --config my_config.yaml
"""

import fire

from spd.postprocess.config import PostprocessConfig


def main(
    wandb_path: str,
    config: str | None = None,
) -> None:
    """Submit all postprocessing jobs for an SPD run.

    Args:
        wandb_path: WandB run path (e.g. wandb:goodfire/spd/runs/abc123).
        config: Path to PostprocessConfig YAML. Uses built-in defaults if omitted.

    Examples:
        spd-postprocess wandb:goodfire/spd/runs/abc123
        spd-postprocess wandb:goodfire/spd/runs/abc123 --config my_config.yaml
    """
    from spd.postprocess import postprocess

    cfg = PostprocessConfig.from_file(config) if config is not None else PostprocessConfig()
    manifest_path = postprocess(wandb_path=wandb_path, config=cfg)
    print(f"\nManifest: {manifest_path}")


def cli() -> None:
    fire.Fire(main)
