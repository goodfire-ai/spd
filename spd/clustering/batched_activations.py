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
from typing import TYPE_CHECKING, override

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.consts import ActivationsTensor, BatchTensor, ComponentLabels
from spd.clustering.dataset import create_dataset_loader
from spd.data import loop_dataloader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName
from spd.utils.distributed_utils import get_device

if TYPE_CHECKING:
    from spd.clustering.clustering_run_config import ClusteringRunConfig


_BATCH_FORMAT: str = "batch_{idx:04}.zip"
_LABELS_FILE: str = "labels.txt"


@dataclass
class ActivationBatch:
    """Single batch of subcomponent activations"""

    activations: ActivationsTensor
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
                activations_np: Float[np.ndarray, "samples n_components"] = np.load(f)
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
        i: int
        for i in range(len(self.batch_paths)):
            expected_name: str = _BATCH_FORMAT.format(idx=i)
            actual_name: str = self.batch_paths[i].name
            assert expected_name == actual_name, (
                f"Expected batch file '{expected_name}', found '{actual_name}'"
            )

        # Load labels from file
        labels_path: Path = batch_dir / _LABELS_FILE
        assert labels_path.exists(), f"Labels file not found: {labels_path}"
        self._labels: ComponentLabels = ComponentLabels(
            labels_path.read_text().strip().splitlines()
        )

    @property
    def labels(self) -> ComponentLabels:
        """Get component labels for all batches."""
        return self._labels

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

    @override
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
        temp_dir: Path = Path(tempfile.mkdtemp(prefix="batch_temp_"))

        # Normalize labels
        normalized_labels: ComponentLabels = ComponentLabels(labels)

        # Save labels file
        labels_path: Path = temp_dir / _LABELS_FILE
        labels_path.write_text("\n".join(normalized_labels))

        # Save the single batch
        batch: ActivationBatch = ActivationBatch(activations=activations, labels=normalized_labels)
        batch.save(temp_dir / _BATCH_FORMAT.format(idx=0))

        # Return BatchedActivations that will cycle through this single batch
        return BatchedActivations(temp_dir)


def _generate_activation_batches(
    model: ComponentModel,
    device: str,
    task_name: TaskName,
    model_path: str,
    batch_size: int,
    n_batches: int,
    output_dir: Path,
    base_seed: int,
    dataset_streaming: bool = False,
) -> None:
    """Core function to generate activation batches.

    Batches are saved WITHOUT filtering - they contain raw/unfiltered activations.
    This is required for merge_iteration to correctly recompute costs from fresh batches.

    Args:
        model: ComponentModel to compute activations
        device: Device to use for computation
        task_name: Task name for dataset loading
        model_path: Path to model for dataset loading (as string)
        batch_size: Batch size for dataset
        n_batches: Number of batches to generate
        output_dir: Directory to save batches
        base_seed: Base seed for dataset loading
        dataset_streaming: Whether to use streaming for dataset loading
    """

    # Create dataloader ONCE instead of reloading for each batch
    dataloader = create_dataset_loader(
        model_path=model_path,
        task_name=task_name,
        batch_size=batch_size,
        seed=base_seed,
        config_kwargs=dict(
            streaming=dataset_streaming,
        ),
    )

    # Use loop_dataloader for efficient iteration that handles exhaustion
    batch_iterator = loop_dataloader(dataloader)

    batch_idx: int
    for batch_idx in tqdm(range(n_batches), desc="Generating batches", leave=False):
        # Get next batch from iterator
        batch_data_raw = next(batch_iterator)

        # Extract input based on task type
        if task_name == "lm":
            batch_data: BatchTensor = batch_data_raw["input_ids"].to(device)
        elif task_name == "resid_mlp":
            batch_data = batch_data_raw[0].to(device)  # (batch, labels) tuple
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        # Compute activations
        with torch.no_grad():
            acts_dict: dict[str, ActivationsTensor] = component_activations(
                model, device, batch_data
            )

        # Process activations WITHOUT filtering
        # Batches must contain raw/unfiltered activations because merge_iteration
        # expects to reload unfiltered data when recomputing costs
        processed: ProcessedActivations = process_activations(
            activations=acts_dict,
            filter_dead_threshold=0.0,  # Never filter when saving batches
            seq_mode="concat" if task_name == "lm" else None,
            filter_modules=None,  # Never filter modules when saving batches
        )

        # Save labels file (once, from first batch)
        if batch_idx == 0:
            labels_path: Path = output_dir / _LABELS_FILE
            labels_path.write_text("\n".join(processed.labels))

        # Save as ActivationBatch
        activation_batch: ActivationBatch = ActivationBatch(
            activations=processed.activations.cpu(),  # Move to CPU for storage
            labels=ComponentLabels(list(processed.labels)),
        )
        activation_batch.save(output_dir / _BATCH_FORMAT.format(idx=batch_idx))

        # Clean up immediately after saving to avoid memory accumulation
        del batch_data, batch_data_raw, acts_dict, processed, activation_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del dataloader, batch_iterator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def precompute_batches_for_single_run(
    clustering_run_config: "ClusteringRunConfig",
    output_dir: Path,
    base_seed: int,
) -> int:
    """
    Precompute activation batches for a single clustering run.

    This loads the model ONCE, calculates how many batches are needed
    (based on recompute_costs_every and n_iters), generates all batches,
    and saves them to disk.

    Batches are saved WITHOUT filtering to ensure merge_iteration can correctly
    recompute costs from fresh batches.

    Args:
        clustering_run_config: Configuration for clustering run
        output_dir: Directory to save batches (will contain batch_0000.zip, batch_0001.zip, etc.)
        base_seed: Base seed for dataset loading

    Returns:
        Number of batches generated
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model to determine number of components
    device: str = get_device()
    spd_run: SPDRunInfo = SPDRunInfo.from_path(clustering_run_config.model_path)
    model: ComponentModel = ComponentModel.from_run_info(spd_run).to(device)
    task_name: TaskName = spd_run.config.task_config.task_name

    # Count total components directly from model (sum C across all component modules)
    n_components: int = sum(comp.C for comp in model.components.values())

    # Calculate number of iterations and batches needed
    n_iters: int = clustering_run_config.merge_config.get_num_iters(n_components)
    recompute_every: int | None = clustering_run_config.merge_config.recompute_costs_every

    n_batches_needed: int
    if recompute_every is None:
        # Single-batch mode: generate 1 batch, reuse for all iterations
        n_batches_needed = 1
        logger.info(f"Single-batch mode: generating 1 batch for {n_iters} iterations")
    else:
        # Multi-batch mode: generate enough batches to cover all iterations
        n_batches_needed = (n_iters + recompute_every - 1) // recompute_every
        logger.info(
            f"Multi-batch mode: generating {n_batches_needed} batches for {n_iters} iterations (recompute_every={recompute_every})"
        )

    # Generate batches (no filtering applied)
    _generate_activation_batches(
        model=model,
        device=device,
        task_name=task_name,
        model_path=clustering_run_config.model_path,
        batch_size=clustering_run_config.batch_size,
        n_batches=n_batches_needed,
        output_dir=output_dir,
        base_seed=base_seed,
        dataset_streaming=clustering_run_config.dataset_streaming,
    )

    # Clean up model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Generated {n_batches_needed} batches and saved to {output_dir}")
    return n_batches_needed


def precompute_batches_for_ensemble(
    clustering_run_config: "ClusteringRunConfig",
    n_runs: int,
    output_dir: Path,
) -> Path | None:
    """
    Precompute activation batches for all runs in ensemble.

    This generates all batches for all runs by calling precompute_batches_for_single_run()
    for each run with a unique seed offset.

    Args:
        clustering_run_config: Configuration for clustering runs
        n_runs: Number of runs in the ensemble
        output_dir: Base directory to save precomputed batches

    Returns:
        Path to base directory containing batches for all runs,
        or None if single-batch mode (recompute_costs_every=None)
    """
    # Check if multi-batch mode
    recompute_every: int | None = clustering_run_config.merge_config.recompute_costs_every
    if recompute_every is None:
        logger.info("Single-batch mode (recompute_costs_every=`None`), skipping precomputation")
        return None

    logger.info("Multi-batch mode detected, precomputing activation batches")

    # Create batches directory
    batches_base_dir: Path = output_dir / "precomputed_batches"
    batches_base_dir.mkdir(exist_ok=True, parents=True)

    # Generate batches for each run
    run_idx: int
    for run_idx in tqdm(range(n_runs), desc="Ensemble runs"):
        run_batch_dir: Path = batches_base_dir / f"run_{run_idx}"
        run_batch_dir.mkdir(exist_ok=True)

        # Use unique seed offset for this run
        run_seed: int = clustering_run_config.dataset_seed + run_idx * 1000

        # Generate all batches for this run
        precompute_batches_for_single_run(
            clustering_run_config=clustering_run_config,
            output_dir=run_batch_dir,
            base_seed=run_seed,
        )

    logger.info(f"All batches precomputed and saved to {batches_base_dir}")
    return batches_base_dir
