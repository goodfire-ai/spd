import functools
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from muutils.dbg import dbg_tensor

from spd.clustering.math.merge_distances import DistancesArray, DistancesMethod
from spd.clustering.merge_run_config import MergeRunConfig
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.clustering.scripts.s1_split_dataset import split_dataset
from spd.clustering.scripts.s3_normalize_histories import normalize_histories
from spd.clustering.scripts.s4_compute_distances import compute_histories_distances
from spd.log import logger
from spd.settings import REPO_ROOT
from spd.utils.cuda_memory_used import cuda_memory_fraction

# pyright: reportUnreachable=false, reportUnnecessaryIsInstance=false

os.environ["WANDB_QUIET"] = "True"

# Delimiter for parsing structured output from s2_run_clustering.py
RESULT_DELIMITER: str = "-" * 50

# TODO: this is super messy
def distribute_clustering(
    config_path: Path,
    data_files: list[Path],
    devices: list[str],
    save_dir: Path,
    cuda_mem_max: float | None = None,
    max_concurrency: int | None = None,
    log_fn: Callable[[str], None] = print,
    log_fn_error: Callable[..., None] = functools.partial(print, file=sys.stderr),  # noqa: B008
) -> list[dict[str, str | None]]:
    n_devices: int = len(devices)
    if n_devices == 0:
        raise ValueError("devices must be non-empty")
    if max_concurrency is None:
        max_concurrency = len(data_files)
    active: list[tuple[subprocess.Popen[str], Path]] = []
    results: list[dict[str, str | None]] = []

    n_files: int = len(data_files)
    try:
        for idx, dataset in enumerate(data_files):
            device: str = devices[idx % n_devices]

            cmd: list[str] = [
                "uv",
                "run",
                "python",
                str(REPO_ROOT / "spd/clustering/scripts/s2_run_clustering.py"),
                "--config",
                str(config_path),
                "--dataset-path",
                str(dataset),
                "--save-dir",
                str(save_dir),
                "--device",
                device,
            ]

            # TODO: this is a hack
            # wait until at least 20% of GPU memory is free
            if cuda_mem_max is not None:
                log_fn_error("")
                while (m := cuda_memory_fraction(device)) > cuda_mem_max:
                    time.sleep(5)
                    log_fn_error(
                        f"GPU memory usage is too high ({m:.2%}), waiting for it to drop below {cuda_mem_max:.2%}...",
                        end="\r",
                    )
                log_fn_error("")

            proc: subprocess.Popen[str] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            active.append((proc, dataset))
            log_fn(
                f"Started clustering {idx + 1}/{n_files} on {device} (pid={proc.pid})\n\t{dataset}"
            )
            if len(active) >= max_concurrency:
                proc_to_wait: subprocess.Popen[str]
                dataset_path: Path
                proc_to_wait, dataset_path = active[0]
                stdout: str
                stderr: str
                stdout, stderr = proc_to_wait.communicate()
                results.append(_parse_clustering_result(stdout, dataset_path, log_fn))
                log_fn(f"Process {proc_to_wait.pid} finished, removing from active list")
                active.pop(0)
        
        for proc, dataset_path in active:
            stdout, stderr = proc.communicate()
            results.append(_parse_clustering_result(stdout, dataset_path, log_fn))
            log_fn(f"Process {proc.pid} finished, removing from active list")
            
    except Exception as e:
        log_fn_error(f"An error occurred: {e}")
        for proc, _ in active:
            proc.kill()
            log_fn_error(f"Killed process {proc.pid} due to error")
        raise e
    
    return results


def _parse_clustering_result(stdout: str, dataset_path: Path, log_fn: Callable[[str], None]) -> dict[str, str | None]:
    """Parse the JSON result from s2_run_clustering.py stdout"""
    import json
    
    # Print the stdout for user visibility
    log_fn(f"Output from {dataset_path.stem}:")
    log_fn(stdout)
    
    # Split by the delimiter and extract JSON
    parts: list[str] = stdout.split(RESULT_DELIMITER)
    assert len(parts) == 3, f"Expected exactly 3 parts when splitting stdout by '{RESULT_DELIMITER}', got {len(parts)}"
    
    json_str: str = parts[1].strip()
    try:
        result: dict[str, str | None] = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON result from {dataset_path.stem}: {e}\nJSON string: {json_str}") from e


def main(
    config: Path | MergeRunConfig,
    base_path: Path = REPO_ROOT / "data/clustering/",
    distances_method: DistancesMethod = "perm_invariant_hamming",
    devices: Sequence[str] | str = "cuda:0",
    max_concurrency: int | None = None,
    plot: bool = True,
):
    # 0. preprocessing
    # ================================================================================
    logger.set_format("console", "terse")

    logger.section("Preprocessing")

    # Load config
    merge_run_config: MergeRunConfig
    config_path: Path
    if isinstance(config, Path):
        merge_run_config = MergeRunConfig.from_file(config)
        config_path = config
    else:
        merge_run_config = config
        config_path = REPO_ROOT / f"data/clustering/configs/{merge_run_config.stable_hash}.json"
        merge_run_config.to_file(config_path)

    # device
    devices_: list[str]
    if isinstance(devices, str):
        devices_ = [devices]
    elif isinstance(devices, list):
        devices_ = devices
    else:
        raise TypeError("devices must be a string or a list of strings")

    # saving some info
    merge_run_config_id: str = merge_run_config.config_identifier
    run_path: Path = base_path / merge_run_config_id
    run_path.mkdir(parents=True, exist_ok=True)
    figures_path: Path = run_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    run_config_path: Path = run_path / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            dict(
                merge_run_config=merge_run_config.model_dump(mode="json"),
                base_path=str(base_path),
                devices=devices_,
                max_concurrency=max_concurrency,
                plot=plot,
                repo_root=str(REPO_ROOT),
                run_id=merge_run_config_id,
                run_path=str(run_path),
            ),
            indent="\t",
        )
    )
    logger.info(f"Run config saved to {run_config_path}")

    batches_path: Path = run_path / "batches"
    batches_config_path: Path = run_path / "batches_config.json"
    histories_path: Path = run_path / "merge_history"

    # 1. tokenize and split the dataset into n_batches of batch_size
    # ================================================================================
    logger.section("Splitting dataset")
    _split_dataset_info_path: Path
    split_dataset_info: dict[str, Any]
    _split_dataset_info_path, split_dataset_info = split_dataset(
        config=merge_run_config,
        base_path=run_path,
        save_file_fmt=f"{batches_path}/batch_{{batch_idx}}.npz",
        cfg_file_fmt=batches_config_path.as_posix(),
    )

    data_files: list[Path] = list(map(Path, split_dataset_info["output_files"]))

    # 2. run the clustering on each batch individually
    # ================================================================================
    logger.section("Distributing clustering")
    clustering_results: list[dict[str, str | None]] = distribute_clustering(
        config_path=config_path,
        data_files=data_files,
        save_dir=histories_path,
        devices=devices_,
        max_concurrency=max_concurrency,
        log_fn=lambda msg: logger.info(f"\x1b[36m[spd-cluster]\x1b[0m {msg}"),
        log_fn_error=lambda msg: logger.error(f"\x1b[31m[spd-cluster:err] {msg}\x1b[0m"),
    )

    # collect histories from the actual results, not by globbing
    histories_input: list[str] | list[Path]
    if merge_run_config.wandb_enabled:
        # Use WandB URLs from the actual results
        wandb_urls: list[str] = []
        for result in clustering_results:
            if result["wandb_url"] is not None:
                wandb_urls.append(result["wandb_url"])
        histories_input = wandb_urls
    else:
        # Use local file paths from the actual results  
        histories_files: list[Path] = []
        for result in clustering_results:
            if result["hist_save_path"] is not None:
                histories_files.append(Path(result["hist_save_path"]))
        histories_input = histories_files

    # Validate that we got results for all expected batches
    expected_batch_count: int = len(data_files)
    actual_batch_count: int = len(clustering_results)
    if actual_batch_count != expected_batch_count:
        logger.error(f"Expected {expected_batch_count} batch results, got {actual_batch_count}")
        raise ValueError(f"Missing batch results: expected {expected_batch_count}, got {actual_batch_count}")
    
    if not histories_input:
        logger.error("No merge histories found in clustering results.")
        raise FileNotFoundError(f"No merge histories found in results: {clustering_results=}")
    
    # Log what we collected
    logger.info(f"Successfully collected {len(histories_input)} merge histories from batch results")
    if merge_run_config.wandb_enabled:
        logger.info("Using WandB URLs for history collection")
    else:
        logger.info("Using local file paths for history collection")

    # 3. normalize histories to account for different active components
    # ================================================================================
    logger.section("Normalizing histories")

    merged_hists: dict[str, Any] = normalize_histories(
        histories=histories_input,
        run_dir=run_path / "distances",
    )

    # 4. compute distances between merge histories
    # ================================================================================
    logger.section("Computing distances between merge histories")
    _dists_path: Path
    distances: DistancesArray
    _dists_path, distances = compute_histories_distances(
        merges_path=merged_hists["paths"]["merge_array"],
        method=distances_method,
    )
    dbg_tensor(distances)

    # ================================================================================
    if plot:
        logger.section("Plotting distances")
        plot_dists_distribution(
            distances=distances,
            mode="points",
            # label="v1"
        )
        plt.legend()
        fig_path: Path = figures_path / f"distances_distribution.{distances_method}.png"
        plt.savefig(fig_path)
        logger.info(f"Saved distances distribution plot to {fig_path}")


def cli():
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run clustering on a dataset using a merge run config"
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
    args: argparse.Namespace = parser.parse_args()

    devices: list[str]
    if args.devices is None:
        import torch

        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    else:
        devices = args.devices.split(",")

    main(
        config=args.config,
        base_path=args.base_path,
        devices=devices,
        max_concurrency=args.max_concurrency,
    )


if __name__ == "__main__":
    cli()
