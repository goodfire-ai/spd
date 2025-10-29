"""Core computation logic for finding max-activating text samples."""

from typing import Any

import torch
from muutils.spinner import SpinnerContext
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from spd.clustering.dashboard.core import (
    BatchProcessingStorage,
    ClusterId,
    ClusterLabel,
    DashboardData,
    TrackingCriterion,
    compute_cluster_coactivations,
)
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import get_obj_device


def compute_max_activations(
    model: ComponentModel,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader[Any],
    merge_history: MergeHistory,
    iteration: int,
    n_samples: int,
    n_batches: int,
    clustering_run: str,
) -> DashboardData:
    """Compute max-activating text samples for each cluster and coactivation matrix.

    Args:
        model: ComponentModel to get activations from
        sigmoid_type: Sigmoid type for activation computation
        tokenizer: Tokenizer for decoding text
        dataloader: DataLoader providing batches
        merge_history: MergeHistory containing cluster information
        iteration: Merge iteration to analyze
        n_samples: Number of top samples to track per cluster
        n_batches: Number of batches to process
        clustering_run: Clustering run identifier

    Returns:
        DashboardData with all cluster data, text samples, and coactivation information
    """
    device: torch.device = get_obj_device(model)

    # Setup: Get cluster info and create ClusterId objects
    unique_MH_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid)
        for cid in unique_MH_cluster_indices
    }

    cluster_id_map: dict[int, ClusterId] = {}
    for idx in unique_MH_cluster_indices:
        components: list[dict[str, Any]] = cluster_components[idx]
        assert components, f"Cluster {idx} has no components"
        cluster_id_map[idx] = ClusterId(
            clustering_run=clustering_run,
            iteration=iteration,
            cluster_label=ClusterLabel(idx),
        )

    criteria: list[TrackingCriterion] = [
        TrackingCriterion(
            property_name="max_activation",
            direction="max",
            n_samples=n_samples,
        )
    ]

    # Initialize storage
    storage: BatchProcessingStorage = BatchProcessingStorage.create(
        cluster_id_map, cluster_components
    )

    # Process batches
    print(f"\nProcessing {n_batches} batches...")
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        with SpinnerContext(
            message=f"  Batch {batch_idx + 1}/{n_batches}",
            format_string="  \r{spinner} ({elapsed_time:.2f}s) {message}{value}",
            update_interval=0.33,
        ):
            storage.process_batch(
                batch_data=batch_data,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

    # Create DashboardData and add clusters incrementally
    dashboard: DashboardData = DashboardData.create(text_samples=storage.text_samples)

    for cluster_idx in tqdm(unique_MH_cluster_indices, desc="Building cluster data"):
        cluster_id: ClusterId = cluster_id_map[cluster_idx]
        cluster_hash = cluster_id.to_string()

        dashboard.add_cluster(
            cluster_id=cluster_id,
            cluster_components=cluster_components[cluster_idx],
            criteria=criteria,
            cluster_activations=storage.cluster_activations[cluster_hash],
            cluster_text_hashes=storage.cluster_text_hashes[cluster_hash],
            cluster_tokens=storage.cluster_tokens[cluster_hash],
        )

    # Compute coactivations
    with SpinnerContext(message="Computing cluster coactivations"):
        coactivations, cluster_indices = compute_cluster_coactivations(
            storage.all_cluster_activations
        )

    # Attach coactivation data to dashboard
    dashboard.coactivations = coactivations
    dashboard.cluster_indices = cluster_indices

    return dashboard
