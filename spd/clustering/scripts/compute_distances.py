import json
from pathlib import Path
from typing import Any

import numpy as np
from zanj import ZANJ

from spd.clustering.merge import MergeConfig, MergeHistory, MergeHistoryEnsemble
from spd.settings import REPO_ROOT


def load_merge_histories(
    path: list[Path] | str,
) -> tuple[list[Path], MergeHistoryEnsemble]:
    """Load merge histories from a list of paths or a path format with wildcards"""
    paths_: list[Path]
    if isinstance(path, str):
        paths_ = list(Path(path).glob("*.zanj"))
    elif isinstance(path, list):
        paths_ = path

    data: list[MergeHistory] = [ZANJ().read(p) for p in paths_]
    print(data)
    ensemble: MergeHistoryEnsemble = MergeHistoryEnsemble(data=data)
    return paths_, ensemble


def compute_histories_distances(
    histories: list[Path] | str,
    out_dir: Path = REPO_ROOT / "data/clustering/merge_history/",
) -> tuple[Path, dict[str, Any], np.ndarray]:
    """Main function to load merge histories and compute distances"""
    # get the histories from paths
    ensemble: MergeHistoryEnsemble
    paths: list[Path]
    paths, ensemble = load_merge_histories(histories)

    # save some config info
    config: MergeConfig = ensemble.config
    config_hash: str = config.stable_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path: Path = out_dir / f"config_{config_hash}.json"
    config_data: dict[str, Any] = dict(
        n_iters=ensemble.n_iters,
        n_ensemble=ensemble.n_ensemble,
        c_components=ensemble.c_components,
        merge_config=config.model_dump(mode="json"),
        paths=[str(p) for p in paths],
        repo_root=str(REPO_ROOT),
        shape=ensemble.shape,
    )
    config_path.write_text(json.dumps(config_data, indent="\t"))
    print(f"Config saved to {config_path}")

    # compute the distances
    output_path: Path = out_dir / f"distances_{config_hash}.npz"
    distances = ensemble.get_distances()

    # save distances and return
    np.savez_compressed(
        output_path,
        distances=distances,
    )
    print(f"Distances saved to {output_path}")

    return config_path, config_data, distances
