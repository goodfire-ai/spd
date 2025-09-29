"""Refactored clustering pipeline using separated storage and transformations.

This module demonstrates how to use the storage and transformations layers
to create a clean, testable, and flexible clustering pipeline.
"""

from collections.abc import Callable, Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from torch import Tensor

from spd.clustering.math.merge_distances import compute_distances
from spd.clustering.merge_run_config import RunConfig
from spd.clustering.storage import ClusteringStorage, NormalizedEnsemble
from spd.clustering.transformations import (
    ClusteringOutput,
    EnsembleOutput,
    cluster_batch,
    normalize_histories,
)
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo


class RefactoredClusteringPipeline:
    """Refactored clustering pipeline with separated concerns.

    This class orchestrates the clustering pipeline using:
    - Storage layer for all I/O operations
    - Transformation layer for pure computations
    - Optional logging/monitoring hooks
    """

    def __init__(
        self,
        config: RunConfig,
        storage: ClusteringStorage | None = None,
        log_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Run configuration
            storage: Optional storage object (will create if not provided)
            log_callback: Optional callback for logging progress
        """
        self.config = config

        # Initialize storage if not provided
        if storage is None:
            storage = ClusteringStorage(
                base_path=config.base_path,
                run_identifier=config.config_identifier,
            )
        self.storage = storage

        self.log_callback = log_callback

        # Load model once for reuse
        self._model: ComponentModel | None = None
        self._spd_run: SPDRunInfo | None = None

    def run(self):
        logger.info(f"Starting clustering pipeline for {self.config.config_identifier}")

        self.storage.save_run_config(self.config)

        logger.info("Step 1: Splitting dataset into batches")
        batch_paths = self._split_and_save_dataset()

        logger.info(f"Step 2: Clustering {len(batch_paths)} batches")
        clustering_results = self._process_batches(batch_paths)

        logger.info("Step 3: Normalizing histories into ensemble")
        ensemble_output = self._normalize_and_save(clustering_results)

        logger.info("Step 4: Computing distances")
        _ = self._compute_and_save_distances(ensemble_output)

        logger.info(f"Pipeline completed for {self.config.config_identifier}")

    def _split_and_save_dataset(self) -> list[Path]:
        """Split dataset and save to storage."""
        # Get dataloader based on task
        dataloader, ds_config = self._create_dataloader()

        # Save to storage
        batch_paths = self.storage.save_batches(dataloader, config=ds_config)

        if self.log_callback:
            self.log_callback("dataset_split", {"n_batches": len(batch_paths)})

        return batch_paths

    def _process_batches(self, batch_paths: list[Path]) -> list[ClusteringOutput]:
        """Process batches in parallel using multiprocessing."""
        # Create worker arguments
        worker_args = [
            (
                batch_path,
                self.config,
                self.storage.run_path,
                self.config.devices[i % len(self.config.devices)],
            )
            for i, batch_path in enumerate(batch_paths)
        ]

        # Run in parallel
        n_workers = self.config.workers_per_device * len(self.config.devices)
        with Pool(n_workers) as pool:
            results = pool.map(_parallel_worker, worker_args)

        if self.log_callback:
            self.log_callback("batches_processed", {"n_batches": len(results)})

        return results

    def _normalize_and_save(self, clustering_results: list[ClusteringOutput]) -> EnsembleOutput:
        """Normalize histories and save ensemble."""
        # Extract histories
        histories = [result.history for result in clustering_results]

        # Normalize (pure computation)
        ensemble_output = normalize_histories(histories)

        # Save to storage
        normalized_ensemble = NormalizedEnsemble(
            merge_array=ensemble_output.normalized_merge_array,
            metadata=ensemble_output.metadata,
        )
        self.storage.save_ensemble(normalized_ensemble)

        if self.log_callback:
            self.log_callback(
                "ensemble_created",
                {
                    "n_batches": len(histories),
                    "n_components": ensemble_output.normalized_merge_array.shape[2],
                },
            )

        return ensemble_output

    def _compute_and_save_distances(self, ensemble_output: EnsembleOutput):
        """Compute distances and save to storage."""
        distances = compute_distances(
            normalized_merge_array=ensemble_output.normalized_merge_array,
            method="perm_invariant_hamming",
        )

        self.storage.save_distances(distances, method="perm_invariant_hamming")

        return distances

    def _create_dataloader(self) -> tuple[Iterator[Tensor], dict[str, Any]]:
        """Create dataloader based on task type."""
        from spd.clustering.s1_split_dataset import (
            _get_dataloader_lm,
            _get_dataloader_resid_mlp,
        )

        if self.config.task_name == "lm":
            return _get_dataloader_lm(
                model_path=self.config.model_path,
                batch_size=self.config.batch_size,
            )
        elif self.config.task_name == "resid_mlp":
            return _get_dataloader_resid_mlp(
                model_path=self.config.model_path,
                batch_size=self.config.batch_size,
            )
        else:
            raise ValueError(f"Unknown task name: {self.config.task_name}")


def _parallel_worker(args: tuple[Path, RunConfig, Path, str]) -> ClusteringOutput:
    """Worker function for parallel processing."""
    batch_path, config, storage_path, device = args

    # Create local storage
    storage = ClusteringStorage(base_path=storage_path.parent, run_identifier=storage_path.name)

    # Load model
    spd_run = SPDRunInfo.from_path(config.model_path)
    model = ComponentModel.from_pretrained(spd_run.checkpoint_path).to(device)

    # Load batch
    batch_tensor = storage.load_batch(batch_path)
    batch_id = batch_path.stem

    # Run clustering
    result = cluster_batch(
        model=model,
        batch=batch_tensor,
        merge_config=config.merge_config,
        batch_id=batch_id,
        device=device,
        sigmoid_type=spd_run.config.sigmoid_type,
        task_name=config.task_name,
    )

    # Save history
    storage.save_history(result.history, batch_id)

    return result
