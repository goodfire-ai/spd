"""Storage layer for clustering pipeline - handles all persistence operations."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

from spd.clustering.math.merge_distances import DistancesArray, DistancesMethod, MergesArray
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import RunConfig


@dataclass
class DatasetBatches:
    """Container for dataset batches and their configuration."""

    batches: list[Tensor]
    config: dict[str, Any]


@dataclass
class NormalizedEnsemble:
    """Container for normalized merge array and metadata."""

    merge_array: MergesArray
    metadata: dict[str, Any]


def _write_text_to_path_and_return(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    return path


class ClusteringStorage:
    """Handles all file I/O operations for the clustering pipeline.

    This class provides a clean separation between data transformations and persistence,
    making the pipeline more testable and flexible.
    """

    # Directory structure constants
    DATASET_DIR: str = "dataset"
    BATCHES_DIR: str = "batches"
    HISTORIES_DIR: str = "merge_histories"
    ENSEMBLE_DIR: str = "ensemble"
    DISTANCES_DIR: str = "distances"

    # File naming constants
    RUN_CONFIG_FILE: str = "run_config.json"
    DATASET_CONFIG_FILE: str = "dataset_config.json"
    ENSEMBLE_META_FILE: str = "ensemble_meta.json"
    ENSEMBLE_ARRAY_FILE: str = "ensemble_merge_array.npz"
    BATCH_FILE_FMT: str = "batch_{batch_idx:02d}.npz"
    HISTORY_FILE_FMT: str = "data_{batch_id}"
    MERGE_HISTORY_FILE: str = "merge_history.zip"
    DISTANCES_FILE_FMT: str = "distances.{method}.npz"

    def __init__(self, base_path: Path, run_identifier: str | None = None):
        """Initialize storage with base path and optional run identifier.

        Args:
            base_path: Root directory for all storage operations
            run_identifier: Optional identifier to create a subdirectory for this run
        """
        self.base_path: Path = base_path
        if run_identifier:
            self.run_path = base_path / run_identifier
        else:
            self.run_path = base_path

        # Ensure base directory exists
        self.run_path.mkdir(parents=True, exist_ok=True)

    # directories
    @property
    def dataset_dir(self) -> Path:
        return self.run_path / self.DATASET_DIR

    @property
    def batches_dir(self) -> Path:
        return self.dataset_dir / self.BATCHES_DIR

    @property
    def histories_dir(self) -> Path:
        return self.run_path / self.HISTORIES_DIR

    @property
    def ensemble_dir(self) -> Path:
        return self.run_path / self.ENSEMBLE_DIR

    @property
    def distances_dir(self) -> Path:
        return self.run_path / self.DISTANCES_DIR

    # files
    @property
    def run_config_file(self) -> Path:
        return self.run_path / self.RUN_CONFIG_FILE

    @property
    def dataset_config_file(self) -> Path:
        return self.dataset_dir / self.DATASET_CONFIG_FILE

    @property
    def ensemble_meta_file(self) -> Path:
        return self.ensemble_dir / self.ENSEMBLE_META_FILE

    @property
    def ensemble_array_file(self) -> Path:
        return self.ensemble_dir / self.ENSEMBLE_ARRAY_FILE

    # dynamic

    def batch_path(self, batch_idx: int) -> Path:
        return self.batches_dir / self.BATCH_FILE_FMT.format(batch_idx=batch_idx)

    def history_path(self, batch_id: str) -> Path:
        return (
            self.histories_dir
            / self.HISTORY_FILE_FMT.format(batch_id=batch_id)
            / self.MERGE_HISTORY_FILE
        )

    # Batch storage methods

    def save_dataset_config(self, config: dict[str, Any]) -> Path:
        return _write_text_to_path_and_return(
            self.dataset_config_file, json.dumps(config, indent=2)
        )

    def save_batch(self, batch: Tensor, batch_idx: int) -> Path:
        batch_path: Path = self.batch_path(batch_idx)
        batch_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(batch_path, input_ids=batch.cpu().numpy())
        return batch_path

    def save_batches(self, batches: Iterator[Tensor], config: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []

        self.save_dataset_config(config)

        for idx, batch in enumerate(batches):
            path: Path = self.save_batch(batch, idx)
            paths.append(path)

        return paths

    def load_batch(self, batch_path: Path) -> Int[Tensor, "batch_size n_ctx"]:
        data: dict[str, np.ndarray] = np.load(batch_path)
        return torch.tensor(data["input_ids"])

    def get_batch_paths(self) -> list[Path]:
        return sorted(self.batches_dir.glob("batch_*.npz"))

    # History storage methods

    def save_history(self, history: MergeHistory, batch_id: str) -> Path:
        history_path: Path = self.history_path(batch_id)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.save(history_path)
        return history_path

    def load_history(self, batch_id: str) -> MergeHistory:
        return MergeHistory.read(self.history_path(batch_id))

    def get_history_paths(self) -> list[Path]:
        return sorted(self.histories_dir.glob(f"*/{self.MERGE_HISTORY_FILE}"))

    def load_histories(self) -> list[MergeHistory]:
        return [MergeHistory.read(path) for path in self.get_history_paths()]

    # Ensemble storage methods
    def save_ensemble(self, ensemble: NormalizedEnsemble) -> tuple[Path, Path]:
        """Save normalized ensemble data"""
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path: Path = self.ensemble_meta_file
        metadata_path.write_text(json.dumps(ensemble.metadata, indent=2))

        # Save merge array
        array_path: Path = self.ensemble_array_file
        np.savez_compressed(array_path, merges=ensemble.merge_array)

        return metadata_path, array_path

    def save_distances(self, distances: DistancesArray, method: DistancesMethod) -> Path:
        self.distances_dir.mkdir(parents=True, exist_ok=True)

        distances_path: Path = self.distances_dir / self.DISTANCES_FILE_FMT.format(method=method)
        np.savez_compressed(distances_path, distances=distances)
        return distances_path

    def load_distances(self, method: DistancesMethod) -> DistancesArray:
        distances_path: Path = self.distances_dir / self.DISTANCES_FILE_FMT.format(method=method)
        data: dict[str, np.ndarray] = np.load(distances_path)
        return data["distances"]

    def save_run_config(self, config: RunConfig) -> Path:
        self.run_config_file.write_text(config.model_dump_json(indent=2))
        return self.run_config_file
