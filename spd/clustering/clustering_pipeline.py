"""
Orchestration layer - clean clustering pipeline coordination.

Replaces the original 370+ line subprocess/FD system with simple multiprocessing.Pool.
Each batch loads its own model and WandB run to match original design.
"""

from pathlib import Path

from spd.clustering.merge_run_config import RunConfig

PIPELINE_PATHS: dict[str, str] = {
    "run_record_path": "run_record.json",
    "histories_dir": "merge_histories",
    "dataset_dir": "dataset",
    "ensemble_dir": "ensemble",
    "distances_dir": "distances",
}


class PipelinePaths:
    def __init__(self, config: RunConfig) -> None:
        self.config: RunConfig = config

    @property
    def run_path(self) -> Path:
        return self.config.base_path / self.config.config_identifier

    @property
    def run_record_path(self) -> Path:
        return self.run_path / PIPELINE_PATHS["run_record_path"]

    @property
    def histories_dir(self) -> Path:
        return self.run_path / PIPELINE_PATHS["histories_dir"]

    @property
    def dataset_dir(self) -> Path:
        return self.run_path / PIPELINE_PATHS["dataset_dir"]

    @property
    def ensemble_dir(self) -> Path:
        return self.run_path / PIPELINE_PATHS["ensemble_dir"]

    @property
    def distances_dir(self) -> Path:
        return self.run_path / PIPELINE_PATHS["distances_dir"]


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

    paths: PipelinePaths = PipelinePaths(config=config)

    print(f"Run record saved to {paths.run_record_path}")
    paths.run_record_path.write_text(config.model_dump_json(indent=2))

    print(f"Splitting dataset into {config.n_batches} batches...")
    data_files: list[Path] = split_and_save_dataset(config=config, output_dir=paths.dataset_dir)

    print(
        f"Processing {len(data_files)} batches with {config.workers_per_device} workers per device..."
    )
    results: list[ClusteringResult] = process_batches_parallel(
        data_files=data_files,
        config=config,
        output_dir=paths.histories_dir,
        workers_per_device=config.workers_per_device,
        devices=config.devices,
    )

    normalized_merge_array: MergesArray = normalize_and_save(
        history_paths=[r.history_save_path for r in results],
        output_dir=paths.ensemble_dir,
    )

    distances: DistancesArray = compute_and_save_distances(
        normalized_merge_array=normalized_merge_array,
        output_dir=paths.distances_dir,
    )

    wandb_urls: list[str] = [r.wandb_url for r in results if r.wandb_url]  # Gross - clean up
    create_clustering_report(
        distances=distances,
        method="perm_invariant_hamming",
        wandb_urls=wandb_urls,
        config_identifier=config.config_identifier,
    )
