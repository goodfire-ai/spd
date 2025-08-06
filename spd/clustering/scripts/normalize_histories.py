import json
from pathlib import Path
from typing import Any

import numpy as np
from zanj import ZANJ

from spd.clustering.merge import MergeHistory, MergeHistoryEnsemble, MergesArray
from spd.log import logger
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


def normalize_histories(
    histories: list[Path] | str,
    out_dir: Path = REPO_ROOT / "data/clustering/merge_history/",
) -> dict[str, Any]:
    """Main function to load merge histories and compute distances"""
    # get the histories from paths
    ensemble: MergeHistoryEnsemble
    paths: list[Path]
    paths, ensemble = load_merge_histories(histories)

    # normalize
    normalized_merge_array: MergesArray
    normalized_merge_meta: dict[str, Any]
    normalized_merge_array, normalized_merge_meta = ensemble.normalized()

    # save things
    run_dir: Path = out_dir / f"run_{ensemble.config.stable_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    normalized_merge_meta["paths"] = [str(p) for p in paths]
    normalized_merge_meta["repo_root"] = str(REPO_ROOT)
    path_metadata: Path = run_dir / "ensemble_meta.json"
    path_metadata.write_text(json.dumps(normalized_merge_meta, indent="\t"))
    logger.info(f"metadata saved to {path_metadata}")

    path_merge_arr: Path = run_dir / "ensemble_merge_array.npz"
    np.savez_compressed(
        path_merge_arr,
        merges=normalized_merge_array,
    )
    logger.info(f"merge array saved to {path_merge_arr}")

    path_hist_ensemble: Path = run_dir / "ensemble_raw.zanj"
    ZANJ().save(ensemble, path_hist_ensemble)
    logger.info(f"Ensemble saved to {path_hist_ensemble}")

    return dict(
        ensemble_meta=normalized_merge_meta,
        normalized_merge_array=path_merge_arr,
        ensemble_raw=path_hist_ensemble,
        paths=dict(
            run_dir=run_dir,
            input_histories=paths,
            metadata=path_metadata,
            merge_array=path_merge_arr,
            ensemble_raw=path_hist_ensemble,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Normalize merge histories"
    )
    parser.add_argument(
        "histories",
        type=str,
        help="Path to the merge histories. Should contain just .zanj files for each history",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data/clustering/merge_history/",
        help="Output directory for the normalized histories",
    )

    args: argparse.Namespace = parser.parse_args()
    normalize_histories(args.histories, args.out_dir)
