"""Core computation logic for finding max-activating text samples."""

from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from muutils.spinner import SpinnerContext
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
from spd.utils.general_utils import extract_batch_data, get_module_device


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
) -> DashboardData:
    """Compute max-activating text samples for each cluster.

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
        DashboardData with all cluster data, text samples, and activations
    """
    device: torch.device = get_module_device(model)

    # Get unique cluster indices and component info
    unique_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid)
        for cid in unique_cluster_indices
    }

    # Create ClusterId objects for each cluster
    cluster_id_map: dict[int, ClusterId] = {}
    for idx in unique_cluster_indices:
        components: list[dict[str, Any]] = cluster_components[idx]
        assert components, f"Cluster {idx} has no components"

        cluster_label: ClusterLabel = ClusterLabel(idx)

        cluster_id_map[idx] = ClusterId(
            clustering_run=clustering_run,
            iteration=iteration,
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
        processed: ProcessedActivations = process_activations(activations, seq_mode="concat")

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

        # TODO: no pbar here, outer loop has a pbar
        for cluster_idx in unique_cluster_indices:
            # Compute cluster activations
            acts_2d: Float[Tensor, "batch_size seq_len"] = _compute_cluster_activations(
                processed, cluster_components[cluster_idx], batch_size, seq_len
            )

            if acts_2d.abs().max() == 0:
                continue

            current_cluster_id: ClusterId = cluster_id_map[cluster_idx]
            current_cluster_hash: ClusterIdHash = current_cluster_id.to_string()

            # Batch transfer activations to CPU and convert to numpy
            acts_2d_cpu: Float[np.ndarray, "batch_size seq_len"] = acts_2d.cpu().numpy()

            # Process each sample in batch
            for batch_sample_idx in range(batch_size):
                text_sample = batch_text_samples[batch_sample_idx]
                text_hash = text_sample.text_hash

                # Store activations for this cluster
                activations_np: Float[np.ndarray, " n_ctx"] = acts_2d_cpu[batch_sample_idx]
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

        clusters[cluster_hash] = cluster_data

        # Add to global activations storage
        act_hashes: list[ActivationSampleHash] = activation_batch.activation_hashes
        for i, (text_hash, acts) in enumerate(zip(text_hashes_list, acts_array, strict=True)):
            act_hash: ActivationSampleHash = act_hashes[i]
            activations_map[act_hash] = current_idx
            all_activations_list.append(acts)
            all_text_hashes_list.append(text_hash)
            current_idx += 1

    # TODO: spinner here
    with SpinnerContext(message="Creating combined activations and dashboard data"):
        # Create combined activations batch
        assert all_activations_list, "No activations collected"
        assert cluster_id_map, "No clusters found"

        combined_activations: Float[np.ndarray, "total_samples n_ctx"] = np.stack(all_activations_list)
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

    return dashboard_data
