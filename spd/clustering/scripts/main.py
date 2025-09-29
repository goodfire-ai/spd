import argparse
from pathlib import Path

from spd.clustering.clustering_pipeline import main
from spd.clustering.merge_run_config import MergeRunConfig
from spd.settings import REPO_ROOT


def cli():
    """Command-line interface for clustering."""
    parser = argparse.ArgumentParser(
        description="Run clustering on a dataset using clean architecture"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the merge run config JSON/YAML file",
    )
    # parser.add_argument(
    #     "--base-path",
    #     "-p",
    #     type=Path,
    #     default=REPO_ROOT / ".data/clustering/",
    #     help="Base path for saving clustering outputs",
    # )
    # parser.add_argument(
    #     "--devices",
    #     "-d",
    #     type=str,
    #     default=None,
    #     help="comma-separated list of devices to use for clustering (e.g., 'cuda:0,cuda:1')",
    # )
    # parser.add_argument(
    #     "--workers-per-device",
    #     "-x",
    #     type=int,
    #     default=1,
    #     help="Maximum number of concurrent clustering processes per device (default: 1)",
    # )
    args = parser.parse_args()

    # Parse devices
    if args.devices is None:
        import torch

        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    else:
        devices = args.devices.split(",")

    # TODO override configs with given args

    main(
        config=MergeRunConfig.from_file(args.config),
        base_path=args.base_path,
        devices=devices,
        workers_per_device=args.workers_per_device,
    )


if __name__ == "__main__":
    cli()
