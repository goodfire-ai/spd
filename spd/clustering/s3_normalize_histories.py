import json
from pathlib import Path

import numpy as np
from muutils.dbg import dbg_tensor
from zanj import ZANJ

from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.log import logger
from spd.settings import REPO_ROOT


def normalize_and_ensemble_and_save(
    history_paths: list[Path],
    distances_dir: Path,
) -> Path:
    """Main function to load merge histories and compute distances"""
    # get the histories from paths or URLs

    data = [MergeHistory.read(p) for p in history_paths]
    ensemble = MergeHistoryEnsemble(data=data)

    # normalize
    normalized_merge_array, normalized_merge_meta = ensemble.normalized()
    dbg_tensor(normalized_merge_array)

    # save things
    distances_dir.mkdir(parents=True, exist_ok=True)

    # TODO check this ever gets looked at
    # WARN: history_paths gets set here and also in the paths dict below
    normalized_merge_meta["paths"] = [str(p) for p in history_paths]
    normalized_merge_meta["repo_root"] = str(REPO_ROOT)
    path_metadata: Path = distances_dir / "ensemble_meta.json"
    path_metadata.write_text(json.dumps(normalized_merge_meta, indent="\t"))
    logger.info(f"metadata saved to {path_metadata}")

    enseble_merge_arr_path: Path = distances_dir / "ensemble_merge_array.npz"
    np.savez_compressed(
        enseble_merge_arr_path,
        merges=normalized_merge_array,
    )
    logger.info(f"merge array saved to {enseble_merge_arr_path}")

    path_hist_ensemble: Path = distances_dir / "ensemble_raw.zanj"
    ZANJ().save(ensemble, path_hist_ensemble)
    logger.info(f"Ensemble saved to {path_hist_ensemble}")

    return enseble_merge_arr_path
