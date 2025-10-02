"""CLI script for computing max-activating text samples for language model component clusters."""

import argparse
import json
from pathlib import Path
from typing import Any

import wandb
from muutils.spinner import SpinnerContext
from wandb.apis.public import Run

from spd.clustering.dashboard.compute_max_act import compute_max_activations
from spd.clustering.dashboard.dashboard_io import (
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


def main(
    wandb_run: str,
    output_dir: Path | None,
    iteration: int,
    n_samples: int,
    n_batches: int,
    batch_size: int,
    context_length: int,
) -> None:
    """Compute max-activating text samples for language model component clusters.

    Args:
        wandb_run: WandB clustering run path (e.g., entity/project/run_id)
        output_dir: Base output directory (default: REPO_ROOT/spd/clustering/dashboard/data/)
        iteration: Merge iteration to analyze (negative indexes from end)
        n_samples: Number of top-activating samples to collect per cluster
        n_batches: Number of data batches to process
        batch_size: Batch size for data loading
        context_length: Context length for tokenization
    """
    # Parse wandb run path
    wandb_path: str = wandb_run.removeprefix("wandb:")
    logger.info(f"Loading WandB run: {wandb_path}")

    # Load artifacts from WandB
    merge_history: MergeHistory
    run_config: dict[str, Any]
    with SpinnerContext(message="Loading WandB artifacts"):
        merge_history, run_config = load_wandb_artifacts(wandb_path)

    # Extract run_id for output directory
    api: wandb.Api = wandb.Api()
    run: Run = api.run(wandb_path)
    run_id: str = run.id

    # Get actual iteration number (handle negative indexing)
    actual_iteration: int = (
        iteration if iteration >= 0 else merge_history.n_iters_current + iteration
    )

    # Set up output directory with iteration count
    base_output_dir: Path = output_dir or (REPO_ROOT / "spd/clustering/dashboard/data")
    dir_name: str = f"{run_id}-i{actual_iteration}"
    final_output_dir: Path = base_output_dir / dir_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {final_output_dir}")

    # Setup model and data
    with SpinnerContext(message="Setting up model and data"):
        model, tokenizer, dataloader, config = setup_model_and_data(
            run_config, context_length, batch_size
        )

    # Compute max activations
    logger.info("computing max activations")
    dashboard_data = compute_max_activations(
        model=model,
        sigmoid_type=config.sigmoid_type,
        tokenizer=tokenizer,
        dataloader=dataloader,
        merge_history=merge_history,
        iteration=iteration,
        n_samples=n_samples,
        n_batches=n_batches,
        clustering_run=run_id,
    )
    logger.info(f"computed max activations: {len(dashboard_data.clusters) = }")
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
            tokenizer_name=config.tokenizer_name,  # pyright: ignore[reportArgumentType]
            config_dict=config.model_dump(mode="json"),
            wandb_run_path=run_config["model_path"],
        )

        # Save dashboard data using new structure
        logger.info("Saving dashboard data")
        dashboard_data.save(str(final_output_dir))
        logger.info(f"Dashboard data saved to: {final_output_dir}")

        # Save model info
        model_info_path: Path = final_output_dir / "model_info.json"
        model_info_path.write_text(json.dumps(model_info, indent=2))
        logger.info(f"Model info saved to: {model_info_path}")


def cli() -> None:
    """CLI entry point with argument parsing."""
    logger.info("parsing args")
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute max-activating text samples for language model component clusters."
    )
    parser.add_argument(
        "--wandb-run",
        "-w",
        type=str,
        help="WandB clustering run path (e.g., entity/project/run_id or wandb:entity/project/run_id)",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Base output directory (default: REPO_ROOT/spd/clustering/dashboard/data/)",
        default=(REPO_ROOT / "spd/clustering/dashboard/data"),
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=-1,
        help="Merge iteration to analyze (negative indexes from end, default: -1 for latest)",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=16,
        help="Number of top-activating samples to collect per cluster",
    )
    parser.add_argument(
        "--n-batches",
        "-s",
        type=int,
        default=4,
        help="Number of data batches to process",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--context-length",
        "-c",
        type=int,
        default=64,
        help="Context length for tokenization (default: 64)",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        default=False,
        help="Write bundled HTML files to output directory (default: False)",
    )
    args: argparse.Namespace = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_html:
        write_html_files(args.output_dir)

    main(
        wandb_run=args.wandb_run,
        output_dir=args.output_dir,
        iteration=args.iteration,
        n_samples=args.n_samples,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )



if __name__ == "__main__":
    cli()
