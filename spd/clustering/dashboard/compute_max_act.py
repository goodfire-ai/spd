"""Core computation logic for finding max-activating text samples."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.dashboard.core import (
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterData,
    ClusterId,
    ClusterIdHash,
    ClusterLabel,
    ComponentInfo,
    DashboardData,
    TextSample,
    TextSampleHash,
    TrackingCriterion,
)
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data  # , get_module_device


@dataclass
class ClusterActivations:
    """Vectorized cluster activations for all clusters.

    Attributes:
        activations: Tensor of shape [n_samples, n_clusters] containing cluster activations
        cluster_indices: List mapping column index to cluster index
    """

    activations: Float[Tensor, "n_samples n_clusters"]
    cluster_indices: list[int]


def _compute_cluster_activations(
    processed: ProcessedActivations,
    cluster_components: list[dict[str, Any]],
    batch_size: int,
    seq_len: int,
) -> Float[Tensor, "batch_size seq_len"]:
    """Compute max activations for a cluster across its components.

    Args:
        processed: ProcessedActivations containing all component activations
        cluster_components: List of component info dicts for this cluster
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        2D tensor of cluster activations (batch_size x seq_len)
    """
    # Get indices for components in this cluster
    comp_indices: list[int] = []
    for component_info in cluster_components:
        label: str = component_info["label"]
        comp_idx: int | None = processed.get_label_index(label)
        if comp_idx is not None:
            comp_indices.append(comp_idx)

    if not comp_indices:
        # Return zeros if no valid components
        return torch.zeros((batch_size, seq_len), device=processed.activations.device)

    # Vectorized: max activations across cluster components
    # processed.activations shape: [n_steps, C] where n_steps = batch_size * seq_len
    # Index into component dimension, then take max across components, then reshape
    cluster_acts: Float[Tensor, " n_steps"] = (
        processed.activations[:, comp_indices].max(dim=1).values
    )
    return cluster_acts.view(batch_size, seq_len)


def compute_all_cluster_activations(
    processed: ProcessedActivations,
    cluster_components: dict[int, list[MergeHistory.ClusterComponentInfo]],
    batch_size: int,
    seq_len: int,
) -> ClusterActivations:
    """Compute activations for all clusters in a vectorized manner.

    Args:
        processed: ProcessedActivations containing all component activations
        cluster_components: Dict mapping cluster_idx -> list of component info dicts
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        ClusterActivations with shape [batch_size * seq_len, n_clusters]
    """
    cluster_indices: list[int] = sorted(cluster_components.keys())
    n_clusters: int = len(cluster_indices)
    n_samples: int = batch_size * seq_len
    device: torch.device = processed.activations.device

    # Build cluster activation tensor: [n_samples, n_clusters]
    cluster_acts: Float[Tensor, "n_samples n_clusters"] = torch.zeros(
        (n_samples, n_clusters), device=device
    )

    # For each cluster, compute max activation across its components
    for cluster_col_idx, cluster_idx in enumerate(cluster_indices):
        components: list[MergeHistory.ClusterComponentInfo] = cluster_components[cluster_idx]

        # Get component indices for this cluster
        comp_indices: list[int] = []
        for component_info in components:
            label: str = component_info.label
            comp_idx: int | None = processed.get_label_index(label)
            if comp_idx is not None:
                comp_indices.append(comp_idx)

        if not comp_indices:
            continue

        # Max activation across components for this cluster
        # processed.activations: [n_samples, n_components]
        cluster_acts[:, cluster_col_idx] = processed.activations[:, comp_indices].max(dim=1).values

    return ClusterActivations(activations=cluster_acts, cluster_indices=cluster_indices)


def compute_cluster_coactivations(
    cluster_activations_list: list[ClusterActivations],
) -> tuple[Float[np.ndarray, "n_clusters n_clusters"], list[int]]:
    """Compute coactivation matrix for clusters across all batches.

    Args:
        cluster_activations_list: List of ClusterActivations from each batch

    Returns:
        Tuple of (coactivation_matrix, cluster_indices) where coact[i,j] is the number
        of samples where both cluster i and j activate
    """
    if not cluster_activations_list:
        return np.array([[]], dtype=np.float32), []

    # All batches should have same cluster indices
    cluster_indices: list[int] = cluster_activations_list[0].cluster_indices
    _n_clusters: int = len(cluster_indices)

    # Concatenate all batch activations: [total_samples, n_clusters]
    all_acts: Float[Tensor, "total_samples n_clusters"] = torch.cat(
        [ca.activations for ca in cluster_activations_list], dim=0
    )

    # Binarize activations (1 if cluster activates, 0 otherwise)
    activation_mask: Float[Tensor, "total_samples n_clusters"] = (all_acts > 0).float()

    # Compute coactivation matrix: coact[i,j] = sum over samples of (cluster_i_active * cluster_j_active)
    # Following spd/clustering/merge.py:69
    coact: Float[Tensor, "n_clusters n_clusters"] = activation_mask.T @ activation_mask

    return coact.cpu().numpy(), cluster_indices


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
    device: torch.device = next(model.parameters()).device

    # Resolve iteration (support negative indexes from the end)
    actual_iteration: int = iteration
    if iteration < 0:
        actual_iteration = merge_history.n_iters_current + iteration
    if not (0 <= actual_iteration < merge_history.n_iters_current):
        raise ValueError(
            f"Iteration {iteration} resolved to {actual_iteration}, which is out of bounds"
        )

    # Get unique cluster indices and component info
    unique_cluster_indices: list[int] = merge_history.get_unique_clusters(actual_iteration)
    cluster_components: dict[int, list[MergeHistory.ClusterComponentInfo]] = {
        cid: merge_history.get_cluster_components_info(actual_iteration, cid)
        for cid in unique_cluster_indices
    }

    # Create ClusterId objects for each cluster
    cluster_id_map: dict[int, ClusterId] = {}
    for idx in unique_cluster_indices:
        components: list[MergeHistory.ClusterComponentInfo] = cluster_components[idx]
        assert components, f"Cluster {idx} has no components"

        cluster_label: ClusterLabel = ClusterLabel(idx)

        cluster_id_map[idx] = ClusterId(
            clustering_run=clustering_run,
            iteration=actual_iteration,
            cluster_label=cluster_label,
        )

    # Define tracking criteria
    criteria: list[TrackingCriterion] = [
        TrackingCriterion(
            property_name="max_activation",
            direction="max",
            n_samples=n_samples,
        )
    ]

    # Storage for accumulating activations per cluster
    cluster_activations: dict[ClusterIdHash, list[Float[np.ndarray, " n_ctx"]]] = {
        cluster_id_map[idx].to_string(): [] for idx in unique_cluster_indices
    }
    cluster_text_hashes: dict[ClusterIdHash, list[TextSampleHash]] = {
        cluster_id_map[idx].to_string(): [] for idx in unique_cluster_indices
    }
    cluster_tokens: dict[ClusterIdHash, list[list[str]]] = {
        cluster_id_map[idx].to_string(): [] for idx in unique_cluster_indices
    }
    text_samples: dict[TextSampleHash, TextSample] = {}

    # Storage for computing coactivations
    all_cluster_activations: list[ClusterActivations] = []

    # Process batches
    for batch_idx, batch_data in enumerate(
        tqdm(dataloader, total=n_batches, desc="Processing batches")
    ):
        if batch_idx >= n_batches:
            break

        batch: Int[Tensor, "batch_size n_ctx"] = extract_batch_data(batch_data).to(device)
        batch_size: int
        seq_len: int
        batch_size, seq_len = batch.shape

        # Get activations
        activations: dict[str, Float[Tensor, "n_steps C"]] = component_activations(
            model,
            device,
            batch=batch,
            sigmoid_type=sigmoid_type,
        )
        processed: ProcessedActivations = process_activations(
            activations, seq_mode="concat", filter_dead_threshold=0
        )

        # Batch tokenize all samples once per batch (move to CPU first)
        batch_cpu: Int[Tensor, "batch_size n_ctx"] = batch.cpu()
        batch_tokens_list: list[list[int]] = batch_cpu.tolist()

        # Batch decode full texts
        batch_texts: list[str] = tokenizer.batch_decode(batch_cpu)  # pyright: ignore[reportAttributeAccessIssue]

        # Batch decode individual tokens for all samples
        batch_token_strings: list[list[str]] = [
            tokenizer.batch_decode([[tid] for tid in tokens_list])  # pyright: ignore[reportAttributeAccessIssue]
            for tokens_list in batch_tokens_list
        ]

        # Create text samples for entire batch
        batch_text_samples: list[TextSample] = []
        for text, token_strings in zip(batch_texts, batch_token_strings, strict=True):
            text_sample = TextSample(full_text=text, tokens=token_strings)
            text_hash = text_sample.text_hash
            if text_hash not in text_samples:
                text_samples[text_hash] = text_sample
            batch_text_samples.append(text_sample)

        # Vectorized: compute all cluster activations at once
        cluster_acts: ClusterActivations = compute_all_cluster_activations(
            processed=processed,
            cluster_components=cluster_components,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Store for coactivation computation
        all_cluster_activations.append(cluster_acts)

        # cluster_acts.activations shape: [n_samples, n_clusters] where n_samples = batch_size * seq_len
        # Reshape to [batch_size, seq_len, n_clusters] for easier indexing
        acts_3d: Float[Tensor, "batch_size seq_len n_clusters"] = cluster_acts.activations.view(
            batch_size, seq_len, -1
        )
        acts_3d_cpu: Float[np.ndarray, "batch_size seq_len n_clusters"] = acts_3d.cpu().numpy()

        # Store activations per cluster
        for cluster_col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices):
            # Get activations for this cluster: [batch_size, seq_len]
            cluster_acts_2d: Float[np.ndarray, "batch_size seq_len"] = acts_3d_cpu[
                :, :, cluster_col_idx
            ]

            # Skip if no activations
            if np.abs(cluster_acts_2d).max() == 0:
                continue

            current_cluster_id: ClusterId = cluster_id_map[cluster_idx]
            current_cluster_hash: ClusterIdHash = current_cluster_id.to_string()

            # Store activations for each sample in batch
            for batch_sample_idx in range(batch_size):
                text_sample = batch_text_samples[batch_sample_idx]
                text_hash = text_sample.text_hash

                # Store activations for this cluster
                activations_np: Float[np.ndarray, " n_ctx"] = cluster_acts_2d[batch_sample_idx]
                cluster_activations[current_cluster_hash].append(activations_np)
                cluster_text_hashes[current_cluster_hash].append(text_hash)
                cluster_tokens[current_cluster_hash].append(text_sample.tokens)

    # Build ClusterData for each cluster
    clusters: dict[ClusterIdHash, ClusterData] = {}
    all_activations_list: list[Float[np.ndarray, " n_ctx"]] = []
    all_text_hashes_list: list[TextSampleHash] = []
    activations_map: dict[ActivationSampleHash, int] = {}
    current_idx: int = 0

    # TODO: pbar here
    for cluster_idx in tqdm(unique_cluster_indices, desc="Building cluster data"):
        cluster_id: ClusterId = cluster_id_map[cluster_idx]
        cluster_hash: ClusterIdHash = cluster_id.to_string()

        if not cluster_activations[cluster_hash]:
            continue

        # Stack activations into batch
        acts_array: Float[np.ndarray, "batch n_ctx"] = np.stack(cluster_activations[cluster_hash])
        text_hashes_list: list[TextSampleHash] = cluster_text_hashes[cluster_hash]
        tokens_list: list[list[str]] = cluster_tokens[cluster_hash]

        activation_batch: ActivationSampleBatch = ActivationSampleBatch(
            cluster_id=cluster_id,
            text_hashes=text_hashes_list,
            activations=acts_array,
            tokens=tokens_list,
        )

        # Convert component info to ComponentInfo objects
        components_info: list[ComponentInfo] = [
            ComponentInfo(module=comp.module, index=comp.index)
            for comp in cluster_components[cluster_idx]
        ]

        # Generate ClusterData with stats and top-k samples
        cluster_data: ClusterData = ClusterData.generate(
            cluster_id=cluster_id,
            activation_samples=activation_batch,
            criteria=criteria,
            components=components_info,
        )

        clusters[cluster_hash] = cluster_data

        # Add to global activations storage
        act_hashes: list[ActivationSampleHash] = activation_batch.activation_hashes
        for i, (text_hash, acts) in enumerate(zip(text_hashes_list, acts_array, strict=True)):
            act_hash: ActivationSampleHash = act_hashes[i]
            activations_map[act_hash] = current_idx
            all_activations_list.append(acts)
            all_text_hashes_list.append(text_hash)
            current_idx += 1

    # TODO: spinner here with SpinnerContext(message="Creating combined activations and dashboard data"):
    # Create combined activations batch
    assert all_activations_list, "No activations collected"
    assert cluster_id_map, "No clusters found"

    combined_activations: Float[np.ndarray, "total_samples n_ctx"] = np.stack(
        all_activations_list
    )
    # Use first cluster_id as placeholder since this is for all clusters
    dummy_cluster_id: ClusterId = list(cluster_id_map.values())[0]
    combined_batch: ActivationSampleBatch = ActivationSampleBatch(
        cluster_id=dummy_cluster_id,
        text_hashes=all_text_hashes_list,
        activations=combined_activations,
    )

    # Build DashboardData
    dashboard_data: DashboardData = DashboardData(
        clusters=clusters,
        text_samples=text_samples,
        activations_map=activations_map,
        activations=combined_batch,
    )

    # Compute coactivation matrix
    coactivations: Float[np.ndarray, "n_clusters n_clusters"]
    cluster_indices: list[int]
    coactivations, cluster_indices = compute_cluster_coactivations(all_cluster_activations)

    return dashboard_data, coactivations, cluster_indices
