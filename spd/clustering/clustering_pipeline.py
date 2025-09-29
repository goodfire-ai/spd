"""
Orchestration layer - clean clustering pipeline coordination.

Replaces the original 370+ line subprocess/FD system with simple multiprocessing.Pool.
Each batch loads its own model and WandB run to match original design.
"""

from pathlib import Path

from spd.clustering.merge_run_config import RunConfig


def main(config: RunConfig) -> None:
    from spd.clustering.math.merge_distances import (
        DistancesArray,
        MergesArray,
        compute_and_save_distances,
    )
    from spd.clustering.s1_split_dataset import split_and_save_dataset
    from spd.clustering.s2_clustering import ClusteringResult, process_batches_parallel
    from spd.clustering.s3_normalize_histories import normalize_and_save
    from spd.clustering.s4_compute_distances import create_clustering_report

    # TODO: factor these out into dataclass or something
    run_path: Path = config.base_path / config.config_identifier
    run_record_path: Path = run_path / "run_record.json"
    histories_dir: Path = run_path / "merge_histories"
    dataset_dir: Path = run_path / "dataset"
    ensemble_dir: Path = run_path / "ensemble"
    distances_dir: Path = run_path / "distances"

    print(f"Run record saved to {run_record_path}")
    run_record_path.write_text(config.model_dump_json(indent=2))

    print(f"Splitting dataset into {config.n_batches} batches...")
    data_files: list[Path] = split_and_save_dataset(config=config, output_dir=dataset_dir)

    print(
        f"Processing {len(data_files)} batches with {config.workers_per_device} workers per device..."
    )
    results: list[ClusteringResult] = process_batches_parallel(
        data_files=data_files,
        config=config,
        output_dir=histories_dir,
        workers_per_device=config.workers_per_device,
        devices=config.devices,
    )

    normalized_merge_array: MergesArray = normalize_and_save(
        history_paths=[r.history_save_path for r in results],
        output_dir=ensemble_dir,
    )

    distances: DistancesArray = compute_and_save_distances(
        normalized_merge_array=normalized_merge_array,
        output_dir=distances_dir,
    )

    wandb_urls: list[str] = [r.wandb_url for r in results if r.wandb_url]  # Gross - clean up
    create_clustering_report(
        distances=distances,
        method="perm_invariant_hamming",
        wandb_urls=wandb_urls,
        config_identifier=config.config_identifier,
    )
