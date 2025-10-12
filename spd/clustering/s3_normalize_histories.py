# NOTE: This file is commented out during refactor - the old storage module no longer exists
# TODO: Re-implement this functionality with the new clustering pipeline architecture

# from pathlib import Path
# from typing import Any

# from spd.clustering.consts import MergesArray
# from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
# from spd.clustering.pipeline.storage import ClusteringStorage, NormalizedEnsemble
# from spd.log import logger


# def normalize_and_save(storage: ClusteringStorage) -> MergesArray:
#     """Load merge histories from storage, normalize, and save ensemble"""
#     # load
#     histories: list[MergeHistory] = storage.load_histories()
#     ensemble: MergeHistoryEnsemble = MergeHistoryEnsemble(data=histories)

#     # normalize
#     normalized_merge_array: MergesArray
#     normalized_merge_meta: dict[str, Any]
#     normalized_merge_array, normalized_merge_meta = ensemble.normalized()

#     # save
#     ensemble_data: NormalizedEnsemble = NormalizedEnsemble(
#         merge_array=normalized_merge_array,
#         metadata=normalized_merge_meta,
#     )
#     metadata_path: Path
#     array_path: Path
#     metadata_path, array_path = storage.save_ensemble(ensemble_data)
#     logger.info(f"metadata saved to {metadata_path}")
#     logger.info(f"merge array saved to {array_path}")

#     return normalized_merge_array
