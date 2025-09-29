"""
Orchestration layer - clean clustering pipeline coordination.

Replaces the original 370+ line subprocess/FD system with simple multiprocessing.Pool.
Each batch loads its own model and WandB run to match original design.
"""

from pathlib import Path

from pydantic import BaseModel

from spd.clustering.merge_run_config import RunFilePaths, MergeRunConfig


class RunRecord(BaseModel):
    merge_run_config: MergeRunConfig
    output_dir: Path
    devices: list[str]
    max_concurrency: int
    plot: bool


def main(
    config: MergeRunConfig,
    base_path: Path,
    devices: list[str],
    workers_per_device: int,
):
    """
    The following is (hopefully) correct (thought see there's some repetition I'd like to change)

    base_dir/
        {config.config_identifier}/
            merge_histories/
                {config.config_identifier}-data_{batch_id}/
                    merge_history.zip
                    plots/
                        activations_raw.pdf
                        activations_concat.pdf
                        activations_coact.pdf
                        activations_coact_log.pdf
                        merge_iteration.pdf
            distances/
            figures/
            run_config.json
    """
    from spd.clustering.s1_split_dataset import split_and_save_dataset
    from spd.clustering.s2_clustering import process_batches_parallel
    from spd.clustering.s3_normalize_histories import normalize_and_ensemble_and_save
    from spd.clustering.s4_compute_distances import (
        compute_and_save_distances_new,
        create_clustering_report,
    )

    run_path = base_path / config.config_identifier
    histories_path = run_path / "merge_histories"
    dataset_dir = run_path / "dataset"
    distances_dir = run_path / "distances"
    run_config_path = run_path / "run_config.json"

    print(f"Run config saved to {run_config_path}")
    run_config_path.write_text(config.model_dump_json(indent=2))

    print(f"Splitting dataset into {config.n_batches} batches...")
    data_files = split_and_save_dataset(
        config=config,
        output_dir=dataset_dir,
        save_file_fmt="batch_{batch_idx}.npz",
        cfg_file_fmt="config.json",  # just a place we save a raw dict of metadata
    )

    print(f"Processing {len(data_files)} batches with {workers_per_device} workers per device...")
    results = process_batches_parallel(
        data_files=data_files,
        config=config,
        output_base_dir=histories_path,
        workers_per_device=workers_per_device,
        devices=devices,
    )

    enseble_merge_arr_path = normalize_and_ensemble_and_save(
        history_paths=[r.history_save_path for r in results],
        distances_dir=distances_dir,
    )

    distances = compute_and_save_distances_new(
        merges_path=enseble_merge_arr_path,
        method="perm_invariant_hamming",
    )

    create_clustering_report(
        distances=distances,
        method="perm_invariant_hamming",
        wandb_urls=[r.wandb_url for r in results if r.wandb_url],  # Gross - clean up,
        config_identifier=config.config_identifier,
    )
