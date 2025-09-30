import argparse
from pathlib import Path

from spd.clustering.merge_run_config import RunConfig
from spd.clustering.pipeline.clustering_pipeline import main
from spd.log import logger
from spd.settings import REPO_ROOT


def cli() -> None:
    """Command-line interface for clustering."""

    logger.set_format("console", style="terse")

    parser = argparse.ArgumentParser(
        description="Run clustering on a dataset using clean architecture"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the merge run config JSON/YAML/TOML file",
    )
    parser.add_argument(
        "--base-path",
        "-p",
        type=Path,
        default=REPO_ROOT / ".data/clustering/",
        help="Base path for saving clustering outputs",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        default=None,
        help="Comma-separated list of devices to use for clustering (e.g., 'cuda:0,cuda:1')",
    )
    parser.add_argument(
        "--workers-per-device",
        "-x",
        type=int,
        default=1,
        help="Maximum number of concurrent clustering processes per device (default: 1)",
    )
    args = parser.parse_args()

    logger.info("Starting clustering pipeline")

    # Parse devices
    if args.devices is None:
        import torch

        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        logger.info(f"No devices specified, auto-detected: {devices}")
    else:
        devices = args.devices.split(",")
        logger.info(f"Using specified devices: {devices}")

    # Load and augment config
    # Note that the defaults for args here always override the default values in `RunConfig` itself,
    # but we must have those defaults to avoid type issues
    logger.info(f"Loading config from {args.config}")
    config: RunConfig = RunConfig.read(args.config)
    config.base_path = args.base_path
    config.devices = devices
    config.workers_per_device = args.workers_per_device

    logger.info(f"Configuration loaded: {config.config_identifier}")
    logger.info(f"Base path: {config.base_path}")
    logger.info(f"{config.workers_per_device = }, {config.devices = }, {config.n_batches = }")

    # Run
    main(config=config)

    logger.info("Clustering pipeline completed successfully")


if __name__ == "__main__":
    cli()
