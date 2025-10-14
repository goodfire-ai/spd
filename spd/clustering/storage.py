"""Storage layer for clustering pipeline - handles all persistence operations."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from spd.clustering.consts import BatchTensor, DistancesArray, DistancesMethod, MergesArray
from spd.clustering.merge_run_config import ClusteringRunConfig

if TYPE_CHECKING:
    from spd.clustering.merge_history import MergeHistory


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

    Filesystem structure:
        <base_path>/
        └── <run_identifier>/              # Optional run-specific subdirectory
            ├── run_config.json            # Run configuration parameters
            ├── dataset/                   # Dataset and batch storage
            │   ├── dataset_config.json    # Dataset configuration metadata
            │   └── batches/               # Individual data batches
            │       ├── batch_00.npz       # Batch 0 (input_ids array)
            │       ├── batch_01.npz       # Batch 1
            │       └── ...
            ├── merge_histories/           # Merge history per batch
            │   ├── data_<batch_id>/       # Per-batch history directory
            │   │   └── merge_history.zip  # Compressed merge history
            │   └── ...
            ├── ensemble/                  # Normalized ensemble results
            │   ├── ensemble_meta.json     # Ensemble metadata
            │   └── ensemble_merge_array.npz  # Normalized merge array
            └── distances/                 # Distance matrices
                ├── distances.<method>.npz # Distance array for each method
                └── ...
    """

    # Directory structure constants
    _DATASET_DIR: str = "dataset"
    _BATCHES_DIR: str = "batches"
    _HISTORIES_DIR: str = "merge_histories"
    _ENSEMBLE_DIR: str = "ensemble"
    _DISTANCES_DIR: str = "distances"
    _DASHBOARD_DIR: str = "dashboard"

    # File naming constants
    _RUN_CONFIG_FILE: str = "run_config.json"
    _DATASET_CONFIG_FILE: str = "dataset_config.json"
    _ENSEMBLE_META_FILE: str = "ensemble_meta.json"
    _ENSEMBLE_ARRAY_FILE: str = "ensemble_merge_array.npz"
    _BATCH_FILE_FMT: str = "batch_{batch_idx:02d}.npz"
    _HISTORY_FILE_FMT: str = "{batch_id}"
    _MERGE_HISTORY_FILE: str = "merge_history.zip"
    _DISTANCES_FILE_FMT: str = "distances.{method}.npz"
    _MODEL_INFO_FILE: str = "model_info.json"
    _MAX_ACTIVATIONS_FILE_FMT: str = "max_activations_i{iteration}_n{n_samples}.json"

    def __init__(self, base_path: Path, run_identifier: str | None = None):
        """Initialize storage with base path and optional run identifier.

        Args:
            base_path: Root directory for all storage operations
            run_identifier: Optional identifier to create a subdirectory for this run
        """
        self._base_path: Path = base_path
        if run_identifier:
            self._run_path = base_path / run_identifier
        else:
            self._run_path = base_path

        # Ensure base directory exists
        self._run_path.mkdir(parents=True, exist_ok=True)

    # directories

    # make base and run path properties so we don't accidentally modify them
    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def run_path(self) -> Path:
        return self._run_path

    @property
    def _dataset_dir(self) -> Path:
        return self.run_path / self._DATASET_DIR

    # directories themselves private, use the storage/read methods to interact with them
    @property
    def _batches_dir(self) -> Path:
        return self._dataset_dir / self._BATCHES_DIR

    @property
    def _histories_dir(self) -> Path:
        return self.run_path / self._HISTORIES_DIR

    @property
    def _ensemble_dir(self) -> Path:
        return self.run_path / self._ENSEMBLE_DIR

    @property
    def _distances_dir(self) -> Path:
        return self.run_path / self._DISTANCES_DIR

    @property
    def _dashboard_dir(self) -> Path:
        return self.run_path / self._DASHBOARD_DIR

    # files
    @property
    def run_config_file(self) -> Path:
        return self.run_path / self._RUN_CONFIG_FILE

    @property
    def dataset_config_file(self) -> Path:
        return self._dataset_dir / self._DATASET_CONFIG_FILE

    @property
    def ensemble_meta_file(self) -> Path:
        return self._ensemble_dir / self._ENSEMBLE_META_FILE

    @property
    def ensemble_array_file(self) -> Path:
        return self._ensemble_dir / self._ENSEMBLE_ARRAY_FILE

    @property
    def model_info_file(self) -> Path:
        return self.run_path / self._MODEL_INFO_FILE

    @property
    def dashboard_model_info_file(self) -> Path:
        return self._dashboard_dir / self._MODEL_INFO_FILE

    # dynamic

    def batch_path(self, batch_idx: int) -> Path:
        return self._batches_dir / self._BATCH_FILE_FMT.format(batch_idx=batch_idx)

    def history_path(self, batch_id: str) -> Path:
        return (
            self._histories_dir
            / self._HISTORY_FILE_FMT.format(batch_id=batch_id)
            / self._MERGE_HISTORY_FILE
        )

    def max_activations_path(self, iteration: int, n_samples: int) -> Path:
        return self._dashboard_dir / self._MAX_ACTIVATIONS_FILE_FMT.format(
            iteration=iteration, n_samples=n_samples
        )

    # Batch storage methods

    def save_dataset_config(self, config: dict[str, Any]) -> Path:
        return _write_text_to_path_and_return(
            self.dataset_config_file, json.dumps(config, indent=2)
        )

    def save_batch(self, batch: BatchTensor, batch_idx: int) -> Path:
        batch_path: Path = self.batch_path(batch_idx)
        batch_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(batch_path, input_ids=batch.cpu().numpy())
        return batch_path

    def save_batches(self, batches: Iterator[BatchTensor], config: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []

        self.save_dataset_config(config)

        for idx, batch in enumerate(batches):
            path: Path = self.save_batch(batch, idx)
            paths.append(path)

        return paths

    def load_batch(self, batch_path: Path) -> BatchTensor:
        data: dict[str, np.ndarray] = np.load(batch_path)
        return torch.tensor(data["input_ids"])

    def get_batch_paths(self) -> list[Path]:
        return sorted(self._batches_dir.glob("batch_*.npz"))

    # History storage methods

    def save_history(self, history: "MergeHistory", batch_id: str) -> Path:
        history_path: Path = self.history_path(batch_id)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.save(history_path)
        return history_path

    def load_history(self, batch_id: str) -> "MergeHistory":
        # Import only at runtime to avoid circular dependencies
        from spd.clustering.merge_history import MergeHistory

        return MergeHistory.read(self.history_path(batch_id))

    def get_history_paths(self) -> list[Path]:
        return sorted(self._histories_dir.glob(f"*/{self._MERGE_HISTORY_FILE}"))

    def load_histories(self) -> list["MergeHistory"]:
        # Import only at runtime to avoid circular dependencies
        from spd.clustering.merge_history import MergeHistory

        return [MergeHistory.read(path) for path in self.get_history_paths()]

    # Ensemble related storage methods

    def save_ensemble(self, ensemble: NormalizedEnsemble) -> tuple[Path, Path]:
        """Save normalized ensemble data"""
        self._ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path: Path = self.ensemble_meta_file
        metadata_path.write_text(json.dumps(ensemble.metadata, indent=2))

        # Save merge array
        array_path: Path = self.ensemble_array_file
        np.savez_compressed(array_path, merges=ensemble.merge_array)

        return metadata_path, array_path

    def save_distances(self, distances: DistancesArray, method: DistancesMethod) -> Path:
        self._distances_dir.mkdir(parents=True, exist_ok=True)

        distances_path: Path = self._distances_dir / self._DISTANCES_FILE_FMT.format(method=method)
        np.savez_compressed(distances_path, distances=distances)
        return distances_path

    def load_distances(self, method: DistancesMethod) -> DistancesArray:
        distances_path: Path = self._distances_dir / self._DISTANCES_FILE_FMT.format(method=method)
        data: dict[str, np.ndarray] = np.load(distances_path)
        return data["distances"]

    def save_run_config(self, config: ClusteringRunConfig) -> Path:
        return _write_text_to_path_and_return(
            self.run_config_file, config.model_dump_json(indent=2)
        )

    def load_run_config(self) -> ClusteringRunConfig:
        return ClusteringRunConfig.from_file(self.run_config_file)

    # Dashboard storage methods

    def save_max_activations(
        self, data: dict[int, dict[str, list[dict[str, Any]]]], iteration: int, n_samples: int
    ) -> Path:
        """Save max activations data to dashboard directory."""
        max_act_path: Path = self.max_activations_path(iteration, n_samples)
        return _write_text_to_path_and_return(max_act_path, json.dumps(data, indent=2))

    def save_model_info(self, model_info: dict[str, Any]) -> Path:
        """Save model info to run directory."""
        return _write_text_to_path_and_return(
            self.model_info_file, json.dumps(model_info, indent=2)
        )

    def save_model_info_to_dashboard(self, model_info: dict[str, Any]) -> Path:
        """Save or copy model info to dashboard directory."""
        return _write_text_to_path_and_return(
            self.dashboard_model_info_file, json.dumps(model_info, indent=2)
        )

    def load_model_info(self) -> dict[str, Any] | None:
        """Load model info from run directory if it exists."""
        if self.model_info_file.exists():
            return json.loads(self.model_info_file.read_text())
        return None
