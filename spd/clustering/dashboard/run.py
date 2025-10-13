"""CLI script for computing max-activating text samples for language model component clusters."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from muutils.spinner import SpinnerContext
from pydantic import BaseModel, Field
from wandb.apis.public import Run

from spd.clustering.dashboard.core.compute_max_act import compute_max_activations
from spd.clustering.dashboard.core.dashboard_io import (
    generate_model_info,
    load_wandb_artifacts,
    setup_model_and_data,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.log import logger
from spd.settings import REPO_ROOT


def write_html_files(output_dir: Path) -> None:
    """Write bundled HTML files from _bundled to output directory.

    Args:
        output_dir: Directory to write HTML files to
    """
    import importlib.resources

    # Read bundled HTML files from the _bundled package
    bundled_package = "spd.clustering.dashboard._bundled"

    index_html = importlib.resources.files(bundled_package).joinpath("index.html").read_text()
    cluster_html = importlib.resources.files(bundled_package).joinpath("cluster.html").read_text()

    # Write to output directory
    (output_dir / "index.html").write_text(index_html)
    (output_dir / "cluster.html").write_text(cluster_html)

    logger.info(f"HTML files written to: {output_dir}")


# TODO: BaseModel -> BaseConfig once #200 is merged
class DashboardConfig(BaseModel):
    wandb_run: str = Field(description="WandB clustering run path (e.g., entity/project/run_id)")
    output_dir: Path | None = Field(
        default=None,
        description="Base output directory (default: REPO_ROOT/spd/clustering/dashboard/data/)",
    )
    iteration: int = Field(
        default=-1,
        description="Merge iteration to analyze (negative indexes from end, default: -1 for latest)",
    )
    n_samples: int = Field(
        default=16,
        description="Number of top-activating samples to collect per cluster",
    )
    n_batches: int = Field(
        default=4,
        description="Number of data batches to process",
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for data loading",
    )
    context_length: int = Field(
        default=64,
        description="Context length for tokenization",
    )
    write_html: bool = Field(
        default=False,
        description="Write bundled HTML files to output directory",
    )

    @classmethod
    def read(cls, config_path: Path) -> "DashboardConfig":
        """Load dashboard config from JSON or YAML file.

        Args:
            config_path: Path to config file (.json or .yaml)

        Returns:
            Loaded DashboardConfig
        """
        import yaml

        if config_path.suffix == ".json":
            config_dict = json.loads(config_path.read_text())
        elif config_path.suffix in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls.model_validate(config_dict)


def main(config: DashboardConfig) -> None:
    """Compute max-activating text samples for language model component clusters.

    Args:
        config: Dashboard configuration
    """
    # Parse wandb run path
    wandb_clustering_run: str = config.wandb_run.removeprefix("wandb:")
    logger.info(f"Loading WandB run: {wandb_clustering_run}")

    # Load artifacts from WandB
    merge_history: MergeHistory
    run_config: dict[str, Any]
    with SpinnerContext(message="Loading WandB artifacts"):
        merge_history, run_config = load_wandb_artifacts(wandb_clustering_run)

    # Extract run_id for output directory
    api: wandb.Api = wandb.Api()
    run: Run = api.run(wandb_clustering_run)
    run_id: str = run.id

    # Get actual iteration number (handle negative indexing)
    actual_iteration: int = (
        config.iteration
        if config.iteration >= 0
        else merge_history.n_iters_current + config.iteration
    )

    # Set up output directory with iteration count
    base_output_dir: Path = config.output_dir or (REPO_ROOT / "spd/clustering/dashboard/data")
    dir_name: str = f"{run_id}-i{actual_iteration}"
    final_output_dir: Path = base_output_dir / dir_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {final_output_dir}")

    # Setup model and data
    with SpinnerContext(message="Setting up model and data"):
        model, tokenizer, dataloader, spd_config = setup_model_and_data(
            run_config, config.context_length, config.batch_size
        )

    # Compute max activations
    logger.info("computing max activations")
    dashboard_data, coactivations, cluster_indices = compute_max_activations(
        model=model,
        sigmoid_type=spd_config.sigmoid_type,
        tokenizer=tokenizer,
        dataloader=dataloader,
        merge_history=merge_history,
        iteration=config.iteration,
        n_samples=config.n_samples,
        n_batches=config.n_batches,
        clustering_run=run_id,
    )
    logger.info(f"computed max activations: {len(dashboard_data.clusters) = }")
    logger.info(f"computed coactivations: shape={coactivations.shape}")
    merge: GroupMerge = merge_history.merges[actual_iteration]

    # Generate model information and save
    with SpinnerContext(message="Generating and saving dashboard data"):
        logger.info("Generating model information")
        model_info: dict[str, Any] = generate_model_info(
            model=model,
            merge_history=merge_history,
            merge=merge,
            iteration=actual_iteration,
            model_path=run_config["model_path"],
            tokenizer_name=spd_config.tokenizer_name,  # pyright: ignore[reportArgumentType]
            config_dict=spd_config.model_dump(mode="json"),
            wandb_clustering_run=wandb_clustering_run,
        )

        # Save dashboard data using new structure
        logger.info("Saving dashboard data")
        dashboard_data.save(str(final_output_dir))
        logger.info(f"Dashboard data saved to: {final_output_dir}")

        # Save model info
        model_info_path: Path = final_output_dir / "model_info.json"
        model_info_path.write_text(json.dumps(model_info, indent=2))
        logger.info(f"Model info saved to: {model_info_path}")

        # Save coactivation matrix
        coactivations_path: Path = final_output_dir / "coactivations.npz"
        np.savez(
            coactivations_path,
            coactivations=coactivations,
            cluster_indices=np.array(cluster_indices),
        )
        logger.info(f"Coactivations saved to: {coactivations_path}")


def cli() -> None:
    """CLI entry point with argument parsing."""
    logger.info("parsing args")
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute max-activating text samples for language model component clusters."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to dashboard config file (JSON or YAML)",
    )
    args: argparse.Namespace = parser.parse_args()

    # Load config
    config: DashboardConfig = DashboardConfig.read(args.config)
    logger.info(f"Loaded config from: {args.config}")

    # Setup output directory and write HTML if requested
    output_dir: Path = config.output_dir or (REPO_ROOT / "spd/clustering/dashboard/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.write_html:
        write_html_files(output_dir)

    main(config)


if __name__ == "__main__":
    cli()
