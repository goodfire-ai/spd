"""
Orchestration layer - clean clustering pipeline coordination.

Replaces the original 370+ line subprocess/FD system with simple multiprocessing.Pool.
Each batch loads its own model and WandB run to match original design.
"""

import argparse
from pathlib import Path

from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.s1_split_dataset import split_and_save_dataset
from spd.clustering.s2_clustering import process_batches_parallel
from spd.clustering.s3_normalize_histories import normalize_and_ensemble_and_save
from spd.clustering.s4_compute_distances import (
    compute_and_save_distances_new,
    create_clustering_report,
)
from spd.settings import REPO_ROOT


def main(
    config: MergeRunConfig,
    base_path: Path,
    n_workers: int,
    devices: list[str],
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

    output_dir = base_path / config.config_identifier

    histories_path = output_dir / "merge_histories"
    histories_path.mkdir(parents=True, exist_ok=True)

    # figures_path = output_dir / "figures"
    # figures_path.mkdir(parents=True, exist_ok=True)

    distances_dir = output_dir / "distances"
    distances_dir.mkdir(parents=True, exist_ok=True)

    # TODO see if we actually need this
    # run_config_path = output_dir / "run_config.json"
    # run_config_path.write_text(
    #     json.dumps(
    #         dict(merge_run_config=config.model_dump(mode="json"), base_path=str(base_path), devices=devices, max_concurrency=n_workers, plot=True,  # can we remove this?  repo_root=str(REPO_ROOT), run_id=config.config_identifier, run_path=str(output_dir),),
    #         indent="\t",
    #     )
    # )
    # print(f"Run config saved to {run_config_path}")

    print(f"Splitting dataset into {config.n_batches} batches...")
    data_files = split_and_save_dataset(
        config=config,
        output_path=output_dir,
        save_file_fmt="batch_{batch_idx}.npz",
        cfg_file_fmt="config.json",  # just a place we save a raw dict of metadata
    )

    print(f"Processing {len(data_files)} batches with {n_workers} workers...")
    results = process_batches_parallel(
        data_files=data_files,
        config=config,
        output_base_dir=histories_path,
        n_workers=n_workers,
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


def cli():
    """Command-line interface for clustering."""
    parser = argparse.ArgumentParser(
        description="Run clustering on a dataset using clean architecture"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the merge run config JSON/YAML file",
    )
    parser.add_argument(
        "--base-path",
        "-p",
        type=Path,
        default=REPO_ROOT / "data/clustering/",
        help="Base path for saving clustering outputs",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        default=None,
        help="comma-separated list of devices to use for clustering (e.g., 'cuda:0,cuda:1')",
    )
    parser.add_argument(
        "--max-concurrency",
        "-x",
        type=int,
        default=None,
        help="Maximum number of concurrent clustering processes (default: all devices)",
    )
    args = parser.parse_args()

    # Parse devices
    if args.devices is None:
        import torch

        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    else:
        devices = args.devices.split(",")

    main(
        config=MergeRunConfig.from_file(args.config),
        base_path=args.base_path,
        devices=devices,
        n_workers=args.max_concurrency if args.max_concurrency is not None else len(devices),
    )


if __name__ == "__main__":
    cli()
