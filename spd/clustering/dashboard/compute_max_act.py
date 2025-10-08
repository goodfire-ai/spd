"""Core computation logic for finding max-activating text samples."""

from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from muutils.spinner import SpinnerContext
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from spd.clustering.dashboard.core import (
    ActivationSampleBatch,
    BatchProcessingStorage,
    ClusterData,
    ClusterId,
    ClusterIdHash,
    ClusterLabel,
    ComponentActivationData,
    ComponentInfo,
    DashboardData,
    GlobalActivationsAccumulator,
    TextSampleHash,
    TrackingCriterion,
    compute_cluster_coactivations,
)
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import get_obj_device


def _build_cluster_data(
    cluster_idx: int,
    cluster_id_map: dict[int, ClusterId],
    cluster_components: dict[int, list[dict[str, Any]]],
    criteria: list[TrackingCriterion],
    storage: BatchProcessingStorage,
) -> tuple[ClusterData, ActivationSampleBatch] | None:
    """Build ClusterData for a single cluster with component-level data.

    Args:
        cluster_idx: Cluster index
        cluster_id_map: Mapping from cluster indices to ClusterId objects
        cluster_components: Component info for each cluster
        criteria: Tracking criteria for top-k samples
        storage: Storage with accumulated activations

    Returns:
        Tuple of (cluster_data, activation_batch) or None if cluster has no activations
    """
    cluster_id: ClusterId = cluster_id_map[cluster_idx]
    cluster_hash: ClusterIdHash = cluster_id.to_string()

    # Check if cluster has any activations
    if not storage.cluster_activations[cluster_hash]:
        return None

    # Stack cluster-level activations into batch
    acts_array: Float[np.ndarray, "batch n_ctx"] = np.stack(
        storage.cluster_activations[cluster_hash]
    )
    text_hashes_list: list[TextSampleHash] = storage.cluster_text_hashes[cluster_hash]
    tokens_list: list[list[str]] = storage.cluster_tokens[cluster_hash]

    activation_batch: ActivationSampleBatch = ActivationSampleBatch(
        cluster_id=cluster_id,
        text_hashes=text_hashes_list,
        activations=acts_array,
        tokens=tokens_list,
    )

    # Convert component info to ComponentInfo objects
    components_info: list[ComponentInfo] = [
        ComponentInfo(module=comp["module"], index=comp["index"])
        for comp in cluster_components[cluster_idx]
    ]

    # Generate ClusterData with stats and top-k samples
    cluster_data: ClusterData = ClusterData.generate(
        cluster_id=cluster_id,
        activation_samples=activation_batch,
        criteria=criteria,
        components=components_info,
    )

    # Build component-level data
    component_data_dict: dict[str, ComponentActivationData] = {}
    component_labels: list[str] = []

    for comp_info in components_info:
        comp_label: str = comp_info.label
        component_labels.append(comp_label)

        # Get stored component activations
        comp_acts_list: list[Float[np.ndarray, " n_ctx"]] = storage.component_activations[
            cluster_hash
        ][comp_label]

        if not comp_acts_list:
            continue

        # Stack activations
        comp_acts_array: Float[np.ndarray, "n_samples n_ctx"] = np.stack(comp_acts_list)

        # Compute component statistics
        comp_stats: dict[str, Any] = {
            "mean": float(np.mean(comp_acts_array)),
            "max": float(np.max(comp_acts_array)),
            "min": float(np.min(comp_acts_array)),
            "median": float(np.median(comp_acts_array)),
            "n_samples": len(comp_acts_list),
        }

        # Create ComponentActivationData
        # Note: activation_sample_hashes and activation_indices will be filled by GlobalActivationsAccumulator
        component_data_dict[comp_label] = ComponentActivationData(
            component_label=comp_label,
            activation_sample_hashes=[],  # Filled by accumulator
            activation_indices=[],  # Filled by accumulator
            stats=comp_stats,
        )

    # Compute component coactivations and cosine similarities from stored activations
    comp_coactivations: Float[np.ndarray, "n_comps n_comps"] | None = None
    comp_cosine_sims: Float[np.ndarray, "n_comps n_comps"] | None = None

    if component_labels:
        n_comps: int = len(component_labels)
        comp_coactivations = np.zeros((n_comps, n_comps), dtype=np.float32)
        comp_cosine_sims = np.zeros((n_comps, n_comps), dtype=np.float32)

        # Build activation matrix for all components: [n_samples, n_comps]
        comp_act_matrix_list: list[Float[np.ndarray, "n_samples n_ctx"]] = []
        for comp_label in component_labels:
            if comp_label in storage.component_activations[cluster_hash]:
                comp_acts_list = storage.component_activations[cluster_hash][comp_label]
                if comp_acts_list:
                    comp_act_matrix_list.append(np.stack(comp_acts_list))

        if comp_act_matrix_list and len(comp_act_matrix_list) == n_comps:
            # Flatten to [n_samples * n_ctx] per component, then stack
            comp_act_flat: list[Float[np.ndarray, " n_total"]] = [
                arr.flatten() for arr in comp_act_matrix_list
            ]
            comp_act_matrix: Float[np.ndarray, "n_comps n_total"] = np.stack(comp_act_flat, axis=0)

            # Compute coactivations (binarized)
            comp_act_bin: Float[np.ndarray, "n_comps n_total"] = (comp_act_matrix > 0).astype(
                np.float32
            )
            comp_coactivations = comp_act_bin @ comp_act_bin.T

            # Compute cosine similarities
            norms: Float[np.ndarray, " n_comps"] = np.linalg.norm(comp_act_matrix, axis=1)
            norms = np.where(norms > 0, norms, 1.0)
            normalized: Float[np.ndarray, "n_comps n_total"] = (
                comp_act_matrix / norms[:, np.newaxis]
            )
            comp_cosine_sims = normalized @ normalized.T

    # Create updated ClusterData with component-level data
    from dataclasses import replace

    cluster_data = replace(
        cluster_data,
        component_activations=component_data_dict if component_data_dict else None,
        component_coactivations=comp_coactivations,
        component_cosine_similarities=comp_cosine_sims,
    )

    return cluster_data, activation_batch


def compute_max_activations(
    model: ComponentModel,
    sigmoid_type: SigmoidTypes,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader[Any],
    merge_history: MergeHistory,
    iteration: int,
    n_samples: int,
    n_batches: int,
    clustering_run: str,
) -> tuple[DashboardData, Float[np.ndarray, "n_clusters n_clusters"], list[int]]:
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
        Tuple of (DashboardData, coactivation_matrix, cluster_indices) where coactivation_matrix[i,j]
        is the number of samples where both cluster i and j activate, and cluster_indices maps
        matrix positions to cluster IDs
    """
    device: torch.device = get_obj_device(model)

    # Setup: Get cluster info and create ClusterId objects
    unique_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid)
        for cid in unique_cluster_indices
    }

    cluster_id_map: dict[int, ClusterId] = {}
    for idx in unique_cluster_indices:
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
    for batch_idx, batch_data in enumerate(
        tqdm(dataloader, total=n_batches, desc="Processing batches")
    ):
        if batch_idx >= n_batches:
            break

        storage.process_batch(
            batch_data=batch_data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            sigmoid_type=sigmoid_type,
        )

    # Build ClusterData for each cluster
    clusters: dict[ClusterIdHash, ClusterData] = {}
    accumulator: GlobalActivationsAccumulator = GlobalActivationsAccumulator.create()

    for cluster_idx in tqdm(unique_cluster_indices, desc="Building cluster data"):
        result = _build_cluster_data(
            cluster_idx=cluster_idx,
            cluster_id_map=cluster_id_map,
            cluster_components=cluster_components,
            criteria=criteria,
            storage=storage,
        )

        if result is not None:
            cluster_data, activation_batch = result
            clusters[cluster_data.cluster_hash] = cluster_data
            accumulator.add_cluster_data(
                cluster_data,
                activation_batch,
                storage.component_activations,
                storage.component_text_hashes,
            )

    # Finalize: Create DashboardData and compute coactivations
    with SpinnerContext(message="Creating combined activations and dashboard data"):
        assert accumulator.activations_list, "No activations collected"
        assert cluster_id_map, "No clusters found"

        combined_activations: Float[np.ndarray, "total_samples n_ctx"] = np.stack(
            accumulator.activations_list
        )
        dummy_cluster_id: ClusterId = list(cluster_id_map.values())[0]
        combined_batch: ActivationSampleBatch = ActivationSampleBatch(
            cluster_id=dummy_cluster_id,
            text_hashes=accumulator.text_hashes_list,
            activations=combined_activations,
        )

        dashboard_data: DashboardData = DashboardData(
            clusters=clusters,
            text_samples=storage.text_samples,
            activations_map=accumulator.activations_map,
            activations=combined_batch,
        )

        coactivations, cluster_indices = compute_cluster_coactivations(
            storage.all_cluster_activations
        )

    return dashboard_data, coactivations, cluster_indices
