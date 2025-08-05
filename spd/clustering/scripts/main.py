import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spd.clustering.merge import MergeConfig
from spd.clustering.scripts.split_dataset import split_dataset
from spd.settings import REPO_ROOT


# TODO: this is super messy
def distribute_clustering(
    merge_config_path: Path,
    model_path: str,
    data_files: list[Path],
    devices: list[str],
    max_concurrency: int | None = None,
) -> None:
    n_devices: int = len(devices)
    if n_devices == 0:
        raise ValueError("devices must be non-empty")
    if max_concurrency is None:
        max_concurrency = len(data_files)
    active: list[subprocess.Popen[bytes]] = []

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
            "--device",
            device,
        ]
        active.append(subprocess.Popen(cmd))
        if len(active) >= max_concurrency:
            active[0].wait()
            active.pop(0)
    for proc in active:
        proc.wait()


def main(
    merge_config: Path | MergeConfig,
    model_path: str = "wandb:goodfire/spd/runs/ioprgffh",
    base_path: Path = REPO_ROOT / "data/split_datasets",
    n_batches: int = 10,
    batch_size: int = 64,
    devices: Sequence[str] | str = "cuda:0",
    max_concurrency: int | None = None,
):
    # 0. preprocessing
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

    # 1. tokenize and split the dataset into n_batches of batch_size
    split_dataset_info_path: Path
    split_dataset_info: dict[str, Any]
    split_dataset_info_path, split_dataset_info = split_dataset(
        model_path=model_path,
        n_batches=n_batches,
        batch_size=batch_size,
        base_path=base_path,
    )

    data_files: list[Path] = list(map(Path, split_dataset_info["output_files"]))

    # 2. run the clustering on each batch individually
    distribute_clustering(
        merge_config_path=merge_config_path,
        model_path=model_path,
        data_files=data_files,
        devices=devices_,
        max_concurrency=max_concurrency,
    )
