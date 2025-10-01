"""Core computation logic for finding max-activating text samples."""

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
from spd.clustering.dashboard.text_sample import (
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterData,
    ClusterId,
    ClusterIdHash,
    ClusterLabel,
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
    """Compute average activations for a cluster across its components.

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

    # Average activations across cluster components
    cluster_acts: Float[Tensor, " n_steps"] = processed.activations[:, comp_indices].mean(dim=1)
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
    spd_run: str,
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
        spd_run: SPD run identifier
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
        components = cluster_components[idx]
        if components:
            first_label = components[0]["label"]
            module_name, original_index_str = first_label.rsplit(":", 1)
            cluster_label = ClusterLabel(
                module_name=module_name,
                original_index=int(original_index_str),
            )
        else:
            cluster_label = ClusterLabel(module_name="unknown", original_index=idx)

        cluster_id_map[idx] = ClusterId(
            spd_run=spd_run,
            clustering_run=clustering_run,
            iteration=iteration,
            cluster_label=cluster_label,
        )

    # Define tracking criteria
    criteria = [
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
    text_samples: dict[TextSampleHash, TextSample] = {}

    # Process batches
    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches, desc="Processing batches")):
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

        for cluster_idx in unique_cluster_indices:
            # Compute cluster activations
            acts_2d: Float[Tensor, "batch_size seq_len"] = _compute_cluster_activations(
                processed, cluster_components[cluster_idx], batch_size, seq_len
            )

            if acts_2d.abs().max() == 0:
                continue

            cluster_id: ClusterId = cluster_id_map[cluster_idx]
            cluster_hash: ClusterIdHash = cluster_id.to_string()

            # Process each sample in batch
            for batch_idx_i in range(batch_size):
                # Extract tokens and create text sample
                tokens: Int[Tensor, " n_ctx"] = batch[batch_idx_i].cpu()
                tokens_list: list[int] = tokens.tolist()
                text: str = tokenizer.decode(tokens)  # pyright: ignore[reportAttributeAccessIssue]

                token_strings: list[str] = [
                    tokenizer.decode([tid])  # pyright: ignore[reportAttributeAccessIssue]
                    for tid in tokens_list
                ]

                text_sample = TextSample(
                    full_text=text,
                    tokens=token_strings,
                )

                # Store text sample (deduplicated by hash)
                text_hash = text_sample.text_hash
                if text_hash not in text_samples:
                    text_samples[text_hash] = text_sample

                # Store activations for this cluster
                activations_np: Float[np.ndarray, " n_ctx"] = acts_2d[batch_idx_i].cpu().numpy()
                cluster_activations[cluster_hash].append(activations_np)
                cluster_text_hashes[cluster_hash].append(text_hash)

    # Build ClusterData for each cluster
    clusters: dict[ClusterIdHash, ClusterData] = {}
    all_activations_list: list[Float[np.ndarray, " n_ctx"]] = []
    all_text_hashes_list: list[TextSampleHash] = []
    activations_map: dict[ActivationSampleHash, int] = {}  # Maps activation hash to index
    current_idx = 0

    for cluster_idx in unique_cluster_indices:
        cluster_id = cluster_id_map[cluster_idx]
        cluster_hash = cluster_id.to_string()

        if not cluster_activations[cluster_hash]:
            continue

        # Stack activations into batch
        acts_array: Float[np.ndarray, "batch n_ctx"] = np.stack(cluster_activations[cluster_hash])
        text_hashes_list = cluster_text_hashes[cluster_hash]

        activation_batch = ActivationSampleBatch(
            cluster_id=cluster_id,
            text_hashes=text_hashes_list,
            activations=acts_array,
        )

        # Generate ClusterData with stats and top-k samples
        cluster_data = ClusterData.generate(
            cluster_id=cluster_id,
            activation_samples=activation_batch,
            criteria=criteria,
        )

        clusters[cluster_hash] = cluster_data

        # Add to global activations storage
        for i, (text_hash, acts) in enumerate(zip(text_hashes_list, acts_array, strict=True)):
            act_hash = activation_batch.activation_hashes[i]
            activations_map[act_hash] = current_idx
            all_activations_list.append(acts)
            all_text_hashes_list.append(text_hash)
            current_idx += 1

    # Create combined activations batch (using first cluster_id as placeholder)
    if all_activations_list:
        combined_activations = np.stack(all_activations_list)
        # Use a dummy cluster_id since this is for all clusters
        dummy_cluster_id = list(cluster_id_map.values())[0] if cluster_id_map else ClusterId(
            spd_run=spd_run,
            clustering_run=clustering_run,
            iteration=iteration,
            cluster_label=ClusterLabel(module_name="combined", original_index=0),
        )
        combined_batch = ActivationSampleBatch(
            cluster_id=dummy_cluster_id,
            text_hashes=all_text_hashes_list,
            activations=combined_activations,
        )
    else:
        # Empty case
        dummy_cluster_id = ClusterId(
            spd_run=spd_run,
            clustering_run=clustering_run,
            iteration=iteration,
            cluster_label=ClusterLabel(module_name="combined", original_index=0),
        )
        combined_batch = ActivationSampleBatch(
            cluster_id=dummy_cluster_id,
            text_hashes=[],
            activations=np.array([]).reshape(0, 0),
        )

    # Build DashboardData
    dashboard_data = DashboardData(
        clusters=clusters,
        text_samples=text_samples,
        activations_map=activations_map,
        activations=combined_batch,
    )

    return dashboard_data
