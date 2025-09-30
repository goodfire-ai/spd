"""Orchestration layer - clustering pipeline coordination"""

from collections.abc import Iterator
from typing import Any

from spd.clustering.merge_run_config import ClusteringRunConfig
from spd.log import logger


def main(config: ClusteringRunConfig) -> None:
    """Run the complete clustering pipeline.

    Args:
        config: ClusteringRunConfig containing all pipeline parameters
    """
    logger.section("setup")

    from spd.clustering.consts import DistancesArray, DistancesMethod, MergesArray
    from spd.clustering.math.merge_distances import compute_distances
    from spd.clustering.pipeline.s1_split_dataset import BatchTensor, split_dataset
    from spd.clustering.pipeline.s2_clustering import ClusteringResult, process_batches_parallel
    from spd.clustering.pipeline.s3_normalize_histories import normalize_and_save
    from spd.clustering.pipeline.s4_compute_distances import create_clustering_report
    from spd.clustering.pipeline.storage import ClusteringStorage

    logger.info("Imports complete")

    # Initialize storage
    storage: ClusteringStorage = ClusteringStorage(
        base_path=config.base_path, run_identifier=config.config_identifier
    )
    logger.info(f"Initialized storage at: {storage.run_path}")

    # Save run configuration
    storage.save_run_config(config)
    logger.info(f"Run record saved to: {storage.run_config_file}")

    # Split dataset into batches
    logger.info(f"Splitting dataset into {config.n_batches} batches...")
    batches: Iterator[BatchTensor]
    dataset_config: dict[str, Any]
    batches, dataset_config = split_dataset(config=config)
    storage.save_batches(batches=batches, config=dataset_config)
    n_batch_paths: int = len(storage.get_batch_paths())
    logger.info(f"Dataset split complete. Saved {n_batch_paths} batches to: {storage._batches_dir}")

    # Process batches in parallel
    logger.section("computing clusterings")
    logger.info(
        f"Processing {n_batch_paths} batches with {config.workers_per_device} workers per device on {config.devices}..."
    )
    results: list[ClusteringResult] = process_batches_parallel(
        config=config,
        storage=storage,
        workers_per_device=config.workers_per_device,
        devices=config.devices,
    )
    logger.info(f"Batch processing complete. Processed {len(results)} batches")

    logger.section("computing distances")
    # Normalize and save ensemble
    logger.info("Normalizing merge histories across ensemble...")
    normalized_merge_array: MergesArray = normalize_and_save(storage=storage)
    logger.info(
        f"Normalized merge array saved: shape={normalized_merge_array.shape}, dtype={normalized_merge_array.dtype}"
    )

    # Compute distances
    # TODO: read distance method from config
    method: DistancesMethod = "perm_invariant_hamming"
    logger.info(f"Computing distances using method: {method}")
    distances: DistancesArray = compute_distances(
        normalized_merge_array=normalized_merge_array,
        method=method,
    )
    storage.save_distances(distances=distances, method=method)
    logger.info(f"Distances computed and saved: shape={distances.shape}")

    # Create clustering report
    wandb_urls: list[str] = [r.wandb_url for r in results if r.wandb_url]
    logger.info(f"Creating clustering report with {len(wandb_urls)} WandB URLs")
    create_clustering_report(
        distances=distances,
        method=method,
        wandb_urls=wandb_urls,
        config_identifier=config.config_identifier,
    )
    logger.info("Clustering report created successfully")
