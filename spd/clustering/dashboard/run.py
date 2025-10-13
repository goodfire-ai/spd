"""CLI script for computing max-activating text samples for language model component clusters."""

import argparse
import json
from pathlib import Path
from typing import Any

import wandb
from muutils.spinner import SpinnerContext
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from wandb.apis.public import Run

from spd.clustering.dashboard.core import (
    DashboardData,
)
from spd.clustering.dashboard.core.compute_max_act import compute_max_activations
from spd.clustering.dashboard.core.dashboard_config import DashboardConfig
from spd.clustering.dashboard.core.dashboard_io import (
    generate_model_info,
    load_wandb_artifacts,
    setup_model_and_data,
)
from spd.clustering.dashboard.core.util import write_html_files
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel


def main(dashboard_config: DashboardConfig) -> None:
    """Compute max-activating text samples for language model component clusters.

    Args:
        config: Dashboard configuration
    """
    # Parse wandb run path
    wandb_clustering_run: str = dashboard_config.wandb_run.removeprefix("wandb:")
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
        dashboard_config.iteration
        if dashboard_config.iteration >= 0
        else merge_history.n_iters_current + dashboard_config.iteration
    )
    try:
        merge: GroupMerge = merge_history.merges[actual_iteration]
    except Exception as e:
        raise ValueError(
            f"Invalid iteration {dashboard_config.iteration} (resolved to {actual_iteration})"
            f"for merge history with {merge_history.n_iters_current} iterations"
            f"{merge_history.merges.group_idxs.shape = }"
        ) from e

    # Set up output directory with iteration count
    dir_name: str = f"{run_id}-i{actual_iteration}"
    final_output_dir: Path = dashboard_config.output_dir / dir_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {final_output_dir}")

    # Setup model and data
    with SpinnerContext(message="Setting up model and data"):
        model: ComponentModel
        tokenizer: PreTrainedTokenizer
        dataloader: DataLoader[Any]
        spd_config: Config
        model, tokenizer, dataloader, spd_config = setup_model_and_data(
            run_config=run_config,
            context_length=dashboard_config.context_length,
            batch_size=dashboard_config.batch_size,
        )

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

    # Compute max activations
    logger.info("computing max activations")
    dashboard_data: DashboardData = compute_max_activations(
        model=model,
        sigmoid_type=spd_config.sigmoid_type,
        tokenizer=tokenizer,
        dataloader=dataloader,
        merge_history=merge_history,
        iteration=dashboard_config.iteration,
        n_samples=dashboard_config.n_samples,
        n_batches=dashboard_config.n_batches,
        clustering_run=run_id,
    )
    logger.info(f"computed max activations: {len(dashboard_data.clusters) = }")
    if dashboard_data.coactivations is not None:
        logger.info(f"computed coactivations: shape={dashboard_data.coactivations.shape}")

    # Save model info
    model_info_path: Path = final_output_dir / "model_info.json"
    model_info_path.write_text(json.dumps(model_info, indent=2))
    logger.info(f"Model info saved to: {model_info_path}")

    # Save dashboard data
    logger.info("Saving dashboard data...")
    dashboard_data.save(str(final_output_dir))
    logger.info(f"Dashboard data saved to: {final_output_dir}")


def cli() -> None:
    """CLI entry point with argument parsing."""
    # cli
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
    config: DashboardConfig = DashboardConfig.from_file(args.config)
    logger.info(f"Loaded config from: {args.config}")

    # Setup output directory and write HTML if requested
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.write_html:
        write_html_files(config.output_dir)

    # run main logic
    main(config)


if __name__ == "__main__":
    cli()
