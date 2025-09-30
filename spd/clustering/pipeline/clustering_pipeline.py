"""
Orchestration layer - clean clustering pipeline coordination.

Replaces the original 370+ line subprocess/FD system with simple multiprocessing.Pool.
Each batch loads its own model and WandB run to match original design.
"""

from spd.clustering.merge_run_config import RunConfig


def main(config: RunConfig) -> None:
    from spd.clustering.math.merge_distances import (
        DistancesArray,
        MergesArray,
        compute_distances,
    )
    from spd.clustering.pipeline.s1_split_dataset import split_dataset
    from spd.clustering.pipeline.s2_clustering import ClusteringResult, process_batches_parallel
    from spd.clustering.pipeline.s3_normalize_histories import normalize_and_save
    from spd.clustering.pipeline.s4_compute_distances import create_clustering_report
    from spd.clustering.pipeline.storage import ClusteringStorage

    storage = ClusteringStorage(base_path=config.base_path, run_identifier=config.config_identifier)

    print(f"Run record saved to {storage.run_config_file}")
    storage.save_run_config(config)

    print(f"Splitting dataset into {config.n_batches} batches...")
    batches, dataset_config = split_dataset(config=config)
    storage.save_batches(batches=batches, config=dataset_config)

    print(
        f"Processing {storage.get_batch_paths().__len__()} batches with {config.workers_per_device} workers per device..."
    )
    results: list[ClusteringResult] = process_batches_parallel(
        config=config,
        storage=storage,
        workers_per_device=config.workers_per_device,
        devices=config.devices,
    )

    normalized_merge_array: MergesArray = normalize_and_save(storage=storage)

    method = "perm_invariant_hamming"
    distances: DistancesArray = compute_distances(
        normalized_merge_array=normalized_merge_array,
        method=method,
    )
    storage.save_distances(distances=distances, method=method)

    wandb_urls: list[str] = [r.wandb_url for r in results if r.wandb_url]
    create_clustering_report(
        distances=distances,
        method=method,
        wandb_urls=wandb_urls,
        config_identifier=config.config_identifier,
    )
