import json
from pathlib import Path

import numpy as np

# from zanj import ZANJ
from spd.clustering.math.merge_distances import MergesArray
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.log import logger


def normalize_and_save(
    history_paths: list[Path],
    output_dir: Path,
) -> MergesArray:
    """Main function to load merge histories and compute distances"""
    # load
    data = [MergeHistory.read(p) for p in history_paths]
    ensemble = MergeHistoryEnsemble(data=data)

    # normalize
    normalized_merge_array, normalized_merge_meta = ensemble.normalized()

    # save
    output_dir.mkdir(parents=True, exist_ok=True)
    path_metadata: Path = output_dir / "ensemble_meta.json"
    enseble_merge_arr_path: Path = output_dir / "ensemble_merge_array.npz"
    path_metadata.write_text(json.dumps(normalized_merge_meta, indent="\t"))
    logger.info(f"metadata saved to {path_metadata}")
    np.savez_compressed(enseble_merge_arr_path, merges=normalized_merge_array)
    logger.info(f"merge array saved to {enseble_merge_arr_path}")

    # TODO: double check we're already saving everything we need outside of the zanj file

    # path_hist_ensemble: Path = output_dir / "ensemble_raw.zanj"
    # ZANJ().save(ensemble, path_hist_ensemble)
    # logger.info(f"Ensemble saved to {path_hist_ensemble}")

    return normalized_merge_array
