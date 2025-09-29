from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt

from spd.clustering.math.merge_distances import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
    compute_distances,
)
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.log import logger


# def compute_and_save_distances_new(
#     merge_array: MergesArray,
#     method: DistancesMethod = "perm_invariant_hamming",
# ) -> DistancesArray:
#     """Main function to load merge histories and compute distances"""
#     distances: DistancesArray = compute_distances(
#         normalized_merge_array=merge_array,
#         method=method,
#     )

#     distances_path: Path = merges_path.with_suffix(f".{method}.distances.npz")
#     np.savez_compressed(distances_path, distances=distances)

#     return distances


def create_clustering_report(
    distances: DistancesArray,
    method: DistancesMethod,
    wandb_urls: list[str],
    config_identifier: str,
) -> None:
    """Create a WandB report with clustering results and distances plot"""

    # Extract entity/project from first URL for the report
    first_url: str = wandb_urls[0]
    entity: str
    project: str

    if first_url.startswith("wandb:"):
        run_path_parts: list[str] = first_url.replace("wandb:", "").split("/")
        entity, project = run_path_parts[0], run_path_parts[1]
    else:
        # Parse full URL
        parts: list[str] = first_url.split("/")
        if "runs" in parts:
            run_idx: int = parts.index("runs") + 1
            entity, project = parts[run_idx - 3], parts[run_idx - 2]
        else:
            logger.warning(f"Could not parse WandB URL: {first_url}")
            return

    # Initialize WandB run for the summary report
    with wandb.init(
        project=project,
        entity=entity,
        name=f"clustering-summary-{config_identifier}",
        tags=["clustering-summary", f"config:{config_identifier}", f"method:{method}"],
        job_type="clustering-analysis",
        config=dict(config_identifier=config_identifier, method=method),
    ) as run:
        # Create and log the distances distribution plot
        ax = plot_dists_distribution(
            distances=distances, mode="points", label=f"{method} distances"
        )
        plt.title(f"Distance Distribution ({method})")

        # Only add legend if there are labeled artists
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            plt.legend()

        # Get the figure from the axes
        fig = ax.get_figure()

        # Log the plot
        run.log(
            {
                f"distances/{method}": wandb.Image(fig),
                "clustering/config_identifier": config_identifier,
            }
        )

        plt.close(fig)

        # Log metadata about the batch runs
        run.log(
            {
                "batch_runs/urls": wandb_urls,
            }
        )

        # Create a summary table of run information
        run_ids: list[str] = []
        for url in wandb_urls:
            if "runs/" in url:
                run_id = url.split("runs/")[-1]
                run_ids.append(run_id)

        if run_ids:
            run.log({"batch_runs/run_ids": run_ids})

        logger.info(
            f"Created wandb clustering summary report with {len(wandb_urls)} batch runs from config {config_identifier}:\n{run.url}/overview"
        )
