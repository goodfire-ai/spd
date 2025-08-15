import json
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from muutils.dbg import dbg_tensor
from zanj import ZANJ

from spd.clustering.math.merge_distances import MergesArray
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.log import logger
from spd.settings import REPO_ROOT

# pyright: reportUnnecessaryIsInstance=false


# TODO: this is messy, possible duplicates code from elsewhere
def load_merge_histories_from_wandb(
    wandb_urls: list[str],
) -> tuple[list[str], MergeHistoryEnsemble]:
    """Load merge histories from WandB run URLs"""
    api = wandb.Api()
    data: list[MergeHistory] = []

    for url in wandb_urls:
        # Parse URL format: wandb:entity/project/run_id or full URL
        run_path: str
        if url.startswith("wandb:"):
            run_path = url.replace("wandb:", "")
        else:
            # Extract run path from full URL
            # e.g. https://wandb.ai/entity/project/runs/run_id -> entity/project/run_id
            parts: list[str] = url.split("/")
            if "runs" in parts:
                run_idx: int = parts.index("runs") + 1
                run_path = f"{parts[run_idx - 3]}/{parts[run_idx - 2]}/{parts[run_idx]}"
            else:
                raise ValueError(f"Cannot parse WandB URL: {url}")

        run = api.run(run_path)

        # Find and download merge history artifact
        artifacts = run.logged_artifacts()
        merge_history_artifact = None
        for artifact in artifacts:
            if artifact.type == "merge_history":
                merge_history_artifact = artifact
                break

        if merge_history_artifact is None:
            raise ValueError(f"No merge_history artifact found for run {run_path}")

        # Download the artifact using WandB's built-in caching
        artifact_dir: str = merge_history_artifact.download()

        # Find the .zanj file in the downloaded artifact
        zanj_files: list[Path] = list(Path(artifact_dir).glob("*.zanj"))
        if not zanj_files:
            raise ValueError(f"No .zanj file found in artifact for run {run_path}")

        # Load the merge history
        merge_history: MergeHistory = ZANJ().read(zanj_files[0])
        data.append(merge_history)

    ensemble = MergeHistoryEnsemble(data=data)

    return wandb_urls, ensemble


def load_merge_histories(
    path: list[Path] | list[str] | str,
) -> tuple[list[Path] | list[str], MergeHistoryEnsemble]:
    """Load merge histories from a list of paths, WandB URLs, or a path format with wildcards"""
    if isinstance(path, str):
        # Single path with wildcards
        paths_: list[Path] = list(Path(path).glob("*.zanj"))
        data: list[MergeHistory] = [ZANJ().read(p) for p in paths_]
        ensemble: MergeHistoryEnsemble = MergeHistoryEnsemble(data=data)
        return paths_, ensemble
    elif isinstance(path, list):
        if all(isinstance(p, Path) for p in path):
            # List of file paths
            paths_paths: list[Path] = path  # pyright: ignore[reportAssignmentType]
            data = [ZANJ().read(p) for p in paths_paths]
            ensemble = MergeHistoryEnsemble(data=data)
            return paths_paths, ensemble
        elif all(isinstance(p, str) and (p.startswith("wandb:") or "wandb.ai" in p) for p in path):
            # List of WandB URLs
            wandb_urls: list[str] = path  # pyright: ignore[reportAssignmentType]
            return load_merge_histories_from_wandb(wandb_urls)
        else:
            raise ValueError("Mixed or unsupported path types in list")
    else:
        raise ValueError(f"Unsupported path type: {type(path)}")  # pyright: ignore[reportUnreachable]


def normalize_histories(
    histories: list[Path] | list[str] | str,
    run_dir: Path,
) -> dict[str, Any]:
    """Main function to load merge histories and compute distances"""
    # get the histories from paths or URLs
    ensemble: MergeHistoryEnsemble
    paths: list[Path] | list[str]
    paths, ensemble = load_merge_histories(histories)

    # normalize
    normalized_merge_array: MergesArray
    normalized_merge_meta: dict[str, Any]
    normalized_merge_array, normalized_merge_meta = ensemble.normalized()
    dbg_tensor(normalized_merge_array)

    # save things
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
