"""Activation batch storage and precomputation for multi-batch clustering.

This module provides:
1. Data structures for storing and loading activation batches (ActivationBatch, BatchedActivations)
2. Precomputation logic to generate batches for ensemble runs (precompute_batches_for_ensemble)
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from tqdm import tqdm

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.dataset import load_dataset
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device

if TYPE_CHECKING:
    from spd.clustering.clustering_run_config import ClusteringRunConfig


@dataclass
class ActivationBatch:
    """Single batch of activations - just tensors, no processing."""

    activations: Tensor  # [samples, n_components]
    labels: list[str]  # ["module:idx", ...]

    def save(self, path: Path) -> None:
        torch.save({"activations": self.activations, "labels": self.labels}, path)

    @staticmethod
    def load(path: Path) -> "ActivationBatch":
        data = torch.load(path, weights_only=False)
        return ActivationBatch(
            activations=data["activations"],
            labels=data["labels"],
        )


class BatchedActivations:
    """Iterator over activation batches from disk."""

    def __init__(self, batch_dir: Path):
        self.batch_dir = batch_dir
        # Find all batch files: batch_0.pt, batch_1.pt, ...
        self.batch_paths = sorted(batch_dir.glob("batch_*.pt"))
        assert len(self.batch_paths) > 0, f"No batch files found in {batch_dir}"
        self.current_idx = 0

    @property
    def n_batches(self) -> int:
        return len(self.batch_paths)

    def get_next_batch(self) -> ActivationBatch:
        """Load and return next batch, cycling through available batches."""
        batch = ActivationBatch.load(self.batch_paths[self.current_idx])
        self.current_idx = (self.current_idx + 1) % self.n_batches
        return batch


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
    recompute_every = clustering_run_config.merge_config.recompute_costs_every
    if recompute_every == 1:
        logger.info("Single-batch mode (recompute_costs_every=1), skipping precomputation")
        return None

    logger.info("Multi-batch mode detected, precomputing activation batches")

    # Load model to determine number of components
    device = get_device()
    spd_run = SPDRunInfo.from_path(clustering_run_config.model_path)
    model = ComponentModel.from_run_info(spd_run).to(device)
    task_name = spd_run.config.task_config.task_name

    # Get number of components (no filtering, so just count from model)
    # Load a sample to count components
    logger.info("Loading sample batch to count components")
    sample_batch = load_dataset(
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
