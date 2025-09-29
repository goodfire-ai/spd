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


class ClusteringStorage:
    """Handles all file I/O operations for the clustering pipeline.

    This class provides a clean separation between data transformations and persistence,
    making the pipeline more testable and flexible.
    """

    # Directory structure constants
    DATASET_DIR = "dataset"
    BATCHES_DIR = "batches"
    HISTORIES_DIR = "merge_histories"
    ENSEMBLE_DIR = "ensemble"
    DISTANCES_DIR = "distances"

    # File naming constants
    CONFIG_FILE = "dataset_config.json"
    BATCH_FILE_FMT = "batch_{batch_idx:02d}.npz"
    HISTORY_FILE_FMT = "data_{batch_id}"
    MERGE_HISTORY_FILE = "merge_history.zip"
    ENSEMBLE_META_FILE = "ensemble_meta.json"
    ENSEMBLE_ARRAY_FILE = "ensemble_merge_array.npz"
    DISTANCES_FILE_FMT = "distances_{method}.npz"
    RUN_CONFIG_FILE = "run_config.json"

    def __init__(self, base_path: Path, run_identifier: str | None = None):
        """Initialize storage with base path and optional run identifier.

        Args:
            base_path: Root directory for all storage operations
            run_identifier: Optional identifier to create a subdirectory for this run
        """
        self.base_path = base_path
        if run_identifier:
            self.run_path = base_path / run_identifier
        else:
            self.run_path = base_path

        # Ensure base directory exists
        self.run_path.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_dir(self) -> Path:
        """Get dataset directory path."""
        return self.run_path / self.DATASET_DIR

    @property
    def batches_dir(self) -> Path:
        """Get batches directory path."""
        return self.dataset_dir / self.BATCHES_DIR

    @property
    def histories_dir(self) -> Path:
        """Get histories directory path."""
        return self.run_path / self.HISTORIES_DIR

    @property
    def ensemble_dir(self) -> Path:
        """Get ensemble directory path."""
        return self.run_path / self.ENSEMBLE_DIR

    @property
    def distances_dir(self) -> Path:
        """Get distances directory path."""
        return self.run_path / self.DISTANCES_DIR

    # Batch storage methods

    def save_dataset_config(self, config: dict[str, Any]) -> Path:
        """Save dataset configuration to JSON file.

        Args:
            config: Dataset configuration dictionary

        Returns:
            Path to saved configuration file
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.dataset_dir / self.CONFIG_FILE
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    def save_batch(self, batch: Tensor, batch_idx: int) -> Path:
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        batch_path = self.batches_dir / self.BATCH_FILE_FMT.format(batch_idx=batch_idx)

        np.savez_compressed(batch_path, input_ids=batch.cpu().numpy())
        return batch_path

    def save_batches(self, batches: Iterator[Tensor], config: dict[str, Any]) -> list[Path]:
        paths = []

        self.save_dataset_config(config)

        for idx, batch in enumerate(batches):
            path = self.save_batch(batch, idx)
            paths.append(path)

        return paths

    def load_batch(self, batch_path: Path) -> Int[Tensor, "batch_size n_ctx"]:
        """Load a batch from disk.

        Args:
            batch_path: Path to batch file

        Returns:
            Loaded batch tensor
        """
        data = np.load(batch_path)
        return torch.tensor(data["input_ids"])

    def load_batches(self) -> list[Tensor]:
        """Load all batches from the batches directory.

        Returns:
            List of loaded batch tensors
        """
        batch_files = sorted(self.batches_dir.glob("batch_*.npz"))
        return [self.load_batch(path) for path in batch_files]

    def get_batch_paths(self) -> list[Path]:
        """Get sorted list of all batch file paths.

        Returns:
            List of paths to batch files
        """
        return sorted(self.batches_dir.glob("batch_*.npz"))

    # History storage methods

    def save_history(self, history: MergeHistory, batch_id: str) -> Path:
        """Save merge history for a batch.

        Args:
            history: MergeHistory object to save
            batch_id: Identifier for the batch

        Returns:
            Path to saved history file
        """
        history_dir = self.histories_dir / self.HISTORY_FILE_FMT.format(batch_id=batch_id)
        history_dir.mkdir(parents=True, exist_ok=True)

        history_path = history_dir / self.MERGE_HISTORY_FILE
        history.save(history_path)
        return history_path

    def load_history(self, batch_id: str) -> MergeHistory:
        """Load merge history for a batch.

        Args:
            batch_id: Identifier for the batch

        Returns:
            Loaded MergeHistory object
        """
        history_path = (
            self.histories_dir
            / self.HISTORY_FILE_FMT.format(batch_id=batch_id)
            / self.MERGE_HISTORY_FILE
        )
        return MergeHistory.read(history_path)

    def get_history_paths(self) -> list[Path]:
        """Get all history file paths.

        Returns:
            List of paths to history files
        """
        return sorted(self.histories_dir.glob(f"*/{self.MERGE_HISTORY_FILE}"))

    def load_histories(self) -> list[MergeHistory]:
        """Load all merge histories.

        Returns:
            List of loaded MergeHistory objects
        """
        return [MergeHistory.read(path) for path in self.get_history_paths()]

    # Ensemble storage methods

    def save_ensemble(self, ensemble: NormalizedEnsemble) -> tuple[Path, Path]:
        """Save normalized ensemble data.

        Args:
            ensemble: NormalizedEnsemble containing merge array and metadata

        Returns:
            Tuple of (metadata_path, array_path)
        """
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = self.ensemble_dir / self.ENSEMBLE_META_FILE
        metadata_path.write_text(json.dumps(ensemble.metadata, indent=2))

        # Save merge array
        array_path = self.ensemble_dir / self.ENSEMBLE_ARRAY_FILE
        np.savez_compressed(array_path, merges=ensemble.merge_array)

        return metadata_path, array_path

    def save_distances(self, distances: DistancesArray, method: DistancesMethod) -> Path:
        self.distances_dir.mkdir(parents=True, exist_ok=True)

        distances_path = self.distances_dir / self.DISTANCES_FILE_FMT.format(method=method)
        np.savez_compressed(distances_path, distances=distances)
        return distances_path

    def load_distances(self, method: DistancesMethod) -> DistancesArray:
        distances_path = self.distances_dir / self.DISTANCES_FILE_FMT.format(method=method)
        data = np.load(distances_path)
        return data["distances"]

    def save_run_config(self, config: RunConfig) -> Path:
        config_path = self.run
        config_path.write_text(config.model_dump_json(indent=2))
        return config_path
