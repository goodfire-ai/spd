from pathlib import Path
from typing import TypedDict

import fire

from spd.settings import SPD_CACHE_DIR


class ClusteringBatchResult(TypedDict):
    """Result from clustering a single batch."""

    hist_save_path: str
    wandb_url: str | None
    batch_name: str
    config_identifier: str


def main(ensemble_id: str, clustering_runs_dir: Path | None = None) -> None:
    if clustering_runs_dir is None:
        clustering_runs_dir = SPD_CACHE_DIR / "clustering" / "runs"

    run_dirs = [i for i in clustering_runs_dir.iterdir() if i.stem.startswith(ensemble_id)]
    print(run_dirs)

    # Load all runs with the given ensemble_id. These runs will be saved in

    # results: list[ClusteringBatchResult] = distribute_clustering(
    #     config_path=config_path,
    #     data_files=batch_paths,
    #     devices=config.devices,
    #     base_path=config.base_path,
    #     run_identifier=config.config_identifier,
    #     workers_per_device=config.workers_per_device,
    #     log_fn=lambda msg: logger.info(f"{distribute_prefix} {msg}"),
    #     log_fn_error=lambda msg: logger.error(f"{distribute_prefix} {msg}"),
    # )
    # logger.info(f"Batch processing complete. Processed {len(results)} batches")

    # logger.section("computing distances")

    # # Normalize and save ensemble
    # logger.info("Normalizing merge histories across ensemble...")
    # normalized_merge_array: MergesArray = normalize_and_save(storage=storage)
    # logger.info(
    #     f"Normalized merge array saved: shape={normalized_merge_array.shape}, dtype={normalized_merge_array.dtype}"
    # )

    # # Compute distances
    # distances_method: DistancesMethod = config.distances_method
    # logger.info(f"Computing distances using method: {distances_method}")
    # distances: DistancesArray = compute_distances(
    #     normalized_merge_array=normalized_merge_array,
    #     method=distances_method,
    # )
    # storage.save_distances(distances=distances, method=distances_method)
    # logger.info(f"Distances computed and saved: shape={distances.shape}")

    # # Create clustering report
    # wandb_urls: list[str] = [r["wandb_url"] for r in results if r["wandb_url"] is not None]
    # logger.info(f"Creating clustering report with {len(wandb_urls)} WandB URLs")
    # create_clustering_report(
    #     distances=distances,
    #     method=distances_method,
    #     wandb_urls=wandb_urls,
    #     config_identifier=config.config_identifier,
    # )
    # logger.info("Clustering report created successfully")


if __name__ == "__main__":
    fire.Fire(main)
