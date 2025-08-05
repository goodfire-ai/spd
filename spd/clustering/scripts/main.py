import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from spd.clustering.merge import MergeConfig
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.clustering.scripts.compute_distances import compute_histories_distances
from spd.clustering.scripts.split_dataset import split_dataset
from spd.settings import REPO_ROOT


# TODO: this is super messy
def distribute_clustering(
    merge_config_path: Path,
    model_path: str,
    data_files: list[Path],
    devices: list[str],
    save_dir: Path,
    max_concurrency: int | None = None,
) -> None:
    n_devices: int = len(devices)
    if n_devices == 0:
        raise ValueError("devices must be non-empty")
    if max_concurrency is None:
        max_concurrency = len(data_files)
    active: list[subprocess.Popen[bytes]] = []

    n_files: int = len(data_files)
    for idx, dataset in enumerate(data_files):
        device: str = devices[idx % n_devices]

        cmd: list[str] = [
            "uv run python",
            str(REPO_ROOT / "spd/clustering/scripts/run_clustering.py"),
            "--merge-config",
            str(merge_config_path),
            "--model-path",
            model_path,
            "--dataset-path",
            str(dataset),
            "--save-dir",
            str(save_dir),
            "--device",
            device,
        ]
        active.append(subprocess.Popen(cmd))
        print(f"Started clustering for {dataset} on {device} (pid={active[-1].pid}) ({idx + 1}/{n_files})")
        if len(active) >= max_concurrency:
            active[0].wait()
            print(f"Process {active[0].pid} finished, removing from active list")
            active.pop(0)
    for proc in active:
        proc.wait()
        print(f"Process {proc.pid} finished, removing from active list")


def main(
    merge_config: Path | MergeConfig,
    model_path: str = "wandb:goodfire/spd/runs/ioprgffh",
    base_path: Path = REPO_ROOT / "data/clustering/",
    n_batches: int = 10,
    batch_size: int = 64,
    devices: Sequence[str] | str = "cuda:0",
    max_concurrency: int | None = None,
    plot: bool = True,
):
    # 0. preprocessing
    print("Preprocessing...")
    devices_: list[str]
    if isinstance(devices, str):
        devices_ = [devices]
    elif isinstance(devices, list):
        devices_ = devices
    else:
        raise TypeError("devices must be a string or a list of strings")

    merge_config_: MergeConfig
    merge_config_path: Path
    if isinstance(merge_config, Path):
        merge_config_ = MergeConfig.model_validate_json(merge_config.read_text())
        merge_config_path = merge_config
    elif isinstance(merge_config, MergeConfig):
        merge_config_ = merge_config
        merge_config_path = (
            REPO_ROOT / f"data/clustering/merge_history/configs/{merge_config_.stable_hash}.json"
        )
        merge_config_path.write_text(merge_config_.model_dump_json())
    else:
        raise TypeError("merge_config must be a MergeConfig or a Path to a JSON file")
    
    merge_config_hash: str = merge_config_.stable_hash
    merge_run_id: str = f"n{n_batches}_b{batch_size}_{merge_config_hash}"
    run_path: Path = REPO_ROOT / f"data/clustering/{merge_run_id}"
    batches_path: Path = run_path / "batches"
    batches_config_path: Path = run_path / "batches_config.json"
    histories_path: Path = run_path / "merge_history"

    # 1. tokenize and split the dataset into n_batches of batch_size
    print("Splitting dataset...")
    split_dataset_info_path: Path
    split_dataset_info: dict[str, Any]
    split_dataset_info_path, split_dataset_info = split_dataset(
        model_path=model_path,
        n_batches=n_batches,
        batch_size=batch_size,
        base_path=run_path,
        save_file_fmt=f"{batches_path}/batch_{{batch_idx}}.npz",
        cfg_file_fmt=f"{batches_path}/{batches_config_path}",
    )

    data_files: list[Path] = list(map(Path, split_dataset_info["output_files"]))

    # 2. run the clustering on each batch individually
    print("Running clustering...")
    distribute_clustering(
        merge_config_path=merge_config_path,
        model_path=model_path,
        data_files=data_files,
        save_dir=histories_path,
        devices=devices_,
        max_concurrency=max_concurrency,
    )

    histories_files: list[Path] = list(histories_path.glob("*.zanj"))

    # 3. compute the distances between the merge histories
    print("Computing distances...")
    dists_cfg_path: Path
    dists_cfg: dict[str, Any]
    distances: np.ndarray
    dists_cfg_path, dists_cfg, distances = compute_histories_distances(
        histories=histories_files,
        out_dir=run_path / "distances",
    )


    if plot:
        plot_dists_distribution(
            distances=distances,
            mode="points",
            # label="v1"
        )
        plt.legend()
