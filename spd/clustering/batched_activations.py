"""Activation batch storage and precomputation for multi-batch clustering.

This module provides:
1. Data structures for storing and loading activation batches (ActivationBatch, BatchedActivations)
2. Precomputation logic to generate batches for ensemble runs (precompute_batches_for_ensemble)
"""

import gc
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.consts import BatchTensor, ComponentLabels
from spd.clustering.dataset import load_dataset
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName
from spd.utils.distributed_utils import get_device

if TYPE_CHECKING:
    from spd.clustering.clustering_run_config import ClusteringRunConfig


_BATCH_FORMAT: str = "batch_{idx:04}.zip"


@dataclass
class ActivationBatch:
    """Single batch of subcomponent activations"""

    activations: Float[Tensor, "samples n_components"]
    labels: ComponentLabels

    def save(self, path: Path) -> None:
        zf: zipfile.ZipFile
        with zipfile.ZipFile(path, "w") as zf:
            with zf.open("activations.npy", "w") as f:
                np.save(f, self.activations.cpu().numpy())
            zf.writestr("labels.txt", "\n".join(self.labels))

    def save_idx(self, batch_dir: Path, idx: int) -> None:
        self.save(batch_dir / _BATCH_FORMAT.format(idx=idx))

    @staticmethod
    def read(path: Path) -> "ActivationBatch":
        zf: zipfile.ZipFile
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("activations.npy", "r") as f:
                activations_np = np.load(f)
            labels_raw: list[str] = zf.read("labels.txt").decode("utf-8").splitlines()
        return ActivationBatch(
            activations=torch.from_numpy(activations_np),
            labels=ComponentLabels(labels_raw),
        )


class BatchedActivations(Iterator[ActivationBatch]):
    """Iterator over activation batches from disk."""

    def __init__(self, batch_dir: Path):
        self.batch_dir: Path = batch_dir
        # Find all batch files
        _glob_pattern: str = re.sub(r"\{[^{}]*\}", "*", _BATCH_FORMAT)  # returns `batch_*.zip`
        self.batch_paths: list[Path] = sorted(batch_dir.glob(_glob_pattern))
        assert len(self.batch_paths) > 0, f"No batch files found in {batch_dir}"
        self.current_idx: int = 0

        # Verify naming
        for i in range(len(self.batch_paths)):
            expected_name = _BATCH_FORMAT.format(idx=i)
            actual_name = self.batch_paths[i].name
            assert expected_name == actual_name, (
                f"Expected batch file '{expected_name}', found '{actual_name}'"
            )

    @property
    def n_batches(self) -> int:
        return len(self.batch_paths)

    def _get_next_batch(self) -> ActivationBatch:
        """Load and return next batch, cycling through available batches."""
        batch: ActivationBatch = ActivationBatch.read(
            self.batch_paths[self.current_idx % self.n_batches]
        )
        self.current_idx += 1
        return batch

    def __next__(self) -> ActivationBatch:
        return self._get_next_batch()

    @classmethod
    def from_tensor(
        cls, activations: Tensor, labels: ComponentLabels | list[str]
    ) -> "BatchedActivations":
        """Create a BatchedActivations instance from a single activation tensor.

        This is a helper for backward compatibility with tests and code that uses
        single-batch mode. It creates a temporary directory with a single batch file.

        Args:
            activations: Activation tensor [samples, n_components]
            labels: Component labels ["module:idx", ...]

        Returns:
            BatchedActivations instance that cycles through the single batch
        """
        import tempfile

        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="batch_temp_"))

        # Save the single batch
        batch = ActivationBatch(activations=activations, labels=ComponentLabels(labels))
        batch.save(temp_dir / _BATCH_FORMAT.format(idx=0))

        # Return BatchedActivations that will cycle through this single batch
        return BatchedActivations(temp_dir)


def precompute_batches_for_ensemble(
    clustering_run_config: "ClusteringRunConfig",
    n_runs: int,
    output_dir: Path,
) -> Path | None:
    """
    Precompute activation batches for all runs in ensemble.

    This loads the model ONCE and generates all batches for all runs,
    then saves them to disk. Each clustering run will load batches
    from disk without needing the model.

    Args:
        clustering_run_config: Configuration for clustering runs
        n_runs: Number of runs in the ensemble
        output_dir: Base directory to save precomputed batches

    Returns:
        Path to base directory containing batches for all runs,
        or None if single-batch mode (recompute_costs_every=1)
    """
    # Check if multi-batch mode
    recompute_every: int | None = clustering_run_config.merge_config.recompute_costs_every
    if recompute_every is None:
        logger.info("Single-batch mode (recompute_costs_every=`None`), skipping precomputation")
        return None

    logger.info("Multi-batch mode detected, precomputing activation batches")

    # Load model to determine number of components
    device: str = get_device()
    spd_run: SPDRunInfo = SPDRunInfo.from_path(clustering_run_config.model_path)
    model: ComponentModel = ComponentModel.from_run_info(spd_run).to(device)
    task_name: TaskName = spd_run.config.task_config.task_name

    # Get number of components (no filtering, so just count from model)
    # Load a sample to count components
    logger.info("Loading sample batch to count components")
    sample_batch: BatchTensor = load_dataset(
        model_path=clustering_run_config.model_path,
        task_name=task_name,
        batch_size=clustering_run_config.batch_size,
        seed=0,
    ).to(device)

    with torch.no_grad():
        sample_acts = component_activations(model, device, sample_batch)

    # Count total components across all modules
    n_components = sum(act.shape[-1] for act in sample_acts.values())

    # Calculate number of iterations
    n_iters = clustering_run_config.merge_config.get_num_iters(n_components)

    # Calculate batches needed per run
    n_batches_needed = (n_iters + recompute_every - 1) // recompute_every

    logger.info(f"Precomputing {n_batches_needed} batches per run for {n_runs} runs")
    logger.info(f"Total: {n_batches_needed * n_runs} batches")

    # Create batches directory
    batches_base_dir = output_dir / "precomputed_batches"
    batches_base_dir.mkdir(exist_ok=True, parents=True)

    # For each run in ensemble
    for run_idx in tqdm(range(n_runs), desc="Ensemble runs"):
        run_batch_dir = batches_base_dir / f"run_{run_idx}"
        run_batch_dir.mkdir(exist_ok=True)

        # Generate batches for this run
        for batch_idx in tqdm(
            range(n_batches_needed),
            desc=f"  Run {run_idx} batches",
            leave=False,
        ):
            # Use unique seed: base_seed + run_idx * 1000 + batch_idx
            # This ensures different data for each run and each batch
            seed = clustering_run_config.dataset_seed + run_idx * 1000 + batch_idx

            # Load data
            batch_data = load_dataset(
                model_path=clustering_run_config.model_path,
                task_name=task_name,
                batch_size=clustering_run_config.batch_size,
                seed=seed,
            ).to(device)

            # Compute activations
            with torch.no_grad():
                acts_dict = component_activations(model, device, batch_data)

            # Process (concat, NO FILTERING)
            processed = process_activations(
                activations=acts_dict,
                filter_dead_threshold=0.0,  # NO FILTERING
                seq_mode="concat" if task_name == "lm" else None,
                filter_modules=None,
            )

            # Save as ActivationBatch
            activation_batch = ActivationBatch(
                activations=processed.activations.cpu(),  # Move to CPU for storage
                labels=list(processed.labels),
            )
            activation_batch.save(run_batch_dir / f"batch_{batch_idx}.pt")

            # Clean up
            del batch_data, acts_dict, processed, activation_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Clean up model
    del model, sample_batch, sample_acts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"All batches precomputed and saved to {batches_base_dir}")

    return batches_base_dir
