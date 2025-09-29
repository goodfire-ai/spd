"""Pure functional transformations for clustering pipeline.

This module contains all the core computational logic separated from I/O operations.
All functions here are pure - they don't perform any file I/O or have side effects.
"""

from dataclasses import dataclass
from typing import Any, Literal

from jaxtyping import Float
from torch import Tensor

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.math.merge_distances import MergesArray
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes


@dataclass
class BatchData:
    """Container for a single batch of data."""

    data: Tensor
    batch_id: str


@dataclass
class ClusteringOutput:
    """Output from clustering a single batch."""

    history: MergeHistory
    batch_id: str
    activations: ProcessedActivations | None = None


@dataclass
class EnsembleOutput:
    """Output from ensemble normalization."""

    normalized_merge_array: MergesArray
    metadata: dict[str, Any]
    ensemble: MergeHistoryEnsemble


def cluster_batch(
    model: ComponentModel,
    batch: Tensor,
    merge_config: MergeConfig,
    batch_id: str,
    device: str,
    sigmoid_type: SigmoidTypes,
    task_name: str,
) -> ClusteringOutput:
    """Run clustering on a single batch.

    This is the main computational function for processing a batch.

    Args:
        model: ComponentModel to use
        batch: Input batch tensor
        merge_config: Configuration for merge algorithm
        batch_id: Identifier for this batch
        device: Device to run on
        sigmoid_type: Sigmoid type for activations
        task_name: Task name (affects processing)
        return_activations: Whether to return processed activations

    Returns:
        ClusteringOutput with history and optional activations
    """
    # Compute activations
    raw_activations = component_activations(
        model=model,
        batch=batch,
        device=device,
        sigmoid_type=sigmoid_type,
    )

    # Process activations
    processed = process_activations(
        activations=raw_activations,
        filter_dead_threshold=merge_config.filter_dead_threshold,
        seq_mode="concat" if task_name == "lm" else None,
        filter_modules=merge_config.filter_modules,
    )

    # Extract what we need for merge
    activations = processed.activations
    component_labels = processed.labels.copy()

    # Run merge iterations
    history = merge_iteration(
        merge_config=merge_config,
        batch_id=batch_id,
        activations=activations,
        component_labels=component_labels,
        log_callback=None,  # Pure function - no logging
    )

    return ClusteringOutput(history=history, batch_id=batch_id, activations=processed)



def normalize_histories(histories: list[MergeHistory]) -> EnsembleOutput:
    ensemble = MergeHistoryEnsemble(data=histories)
    normalized_merge_array, normalized_meta = ensemble.normalized()
    return EnsembleOutput(
        normalized_merge_array=normalized_merge_array,
        metadata=normalized_meta,
        ensemble=ensemble,
    )
