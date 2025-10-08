"""Core computation logic for finding max-activating text samples."""

from dataclasses import dataclass
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
    ComponentActivationData,
    ComponentInfo,
    DashboardData,
    TextSample,
    TextSampleHash,
    TrackingCriterion,
)
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data, get_obj_device


@dataclass
class ClusterActivations:
    """Vectorized cluster activations for all clusters.

    Attributes:
        activations: Tensor of shape [n_samples, n_clusters] containing cluster activations
        cluster_indices: List mapping column index to cluster index
    """

    activations: Float[Tensor, "n_samples n_clusters"]
    cluster_indices: list[int]


@dataclass
class BatchProcessingStorage:
    """Storage for accumulating activations during batch processing.

    Attributes:
        cluster_activations: Cluster-level activations per cluster
        cluster_text_hashes: Text hashes for cluster-level activations
        cluster_tokens: Token strings for cluster-level activations
        component_activations: Component-level activations per cluster per component
        component_text_hashes: Text hashes for component-level activations
        text_samples: All unique text samples encountered
        all_cluster_activations: Cluster activations for coactivation computation
    """

    cluster_activations: dict[ClusterIdHash, list[Float[np.ndarray, " n_ctx"]]]
    cluster_text_hashes: dict[ClusterIdHash, list[TextSampleHash]]
    cluster_tokens: dict[ClusterIdHash, list[list[str]]]
    component_activations: dict[ClusterIdHash, dict[str, list[Float[np.ndarray, " n_ctx"]]]]
    component_text_hashes: dict[ClusterIdHash, dict[str, list[TextSampleHash]]]
    text_samples: dict[TextSampleHash, TextSample]
    all_cluster_activations: list[ClusterActivations]

    @classmethod
    def create(
        cls,
        cluster_id_map: dict[int, ClusterId],
        cluster_components: dict[int, list[dict[str, Any]]],
    ) -> "BatchProcessingStorage":
        """Create initialized storage structures.

        Args:
            cluster_id_map: Mapping from cluster indices to ClusterId objects
            cluster_components: Component info for each cluster

        Returns:
            Initialized BatchProcessingStorage object
        """
        unique_cluster_indices: list[int] = list(cluster_id_map.keys())

        cluster_activations: dict[ClusterIdHash, list[Float[np.ndarray, " n_ctx"]]] = {
            cluster_id_map[idx].to_string(): [] for idx in unique_cluster_indices
        }
        cluster_text_hashes: dict[ClusterIdHash, list[TextSampleHash]] = {
            cluster_id_map[idx].to_string(): [] for idx in unique_cluster_indices
        }
        cluster_tokens: dict[ClusterIdHash, list[list[str]]] = {
            cluster_id_map[idx].to_string(): [] for idx in unique_cluster_indices
        }
        component_activations: dict[ClusterIdHash, dict[str, list[Float[np.ndarray, " n_ctx"]]]] = {
            cluster_id_map[idx].to_string(): {comp["label"]: [] for comp in cluster_components[idx]}
            for idx in unique_cluster_indices
        }
        component_text_hashes: dict[ClusterIdHash, dict[str, list[TextSampleHash]]] = {
            cluster_id_map[idx].to_string(): {comp["label"]: [] for comp in cluster_components[idx]}
            for idx in unique_cluster_indices
        }
        text_samples: dict[TextSampleHash, TextSample] = {}
        all_cluster_activations: list[ClusterActivations] = []

        return cls(
            cluster_activations=cluster_activations,
            cluster_text_hashes=cluster_text_hashes,
            cluster_tokens=cluster_tokens,
            component_activations=component_activations,
            component_text_hashes=component_text_hashes,
            text_samples=text_samples,
            all_cluster_activations=all_cluster_activations,
        )

    def process_batch(
        self,
        batch_data: Any,
        model: ComponentModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        sigmoid_type: SigmoidTypes,
        cluster_id_map: dict[int, ClusterId],
        cluster_components: dict[int, list[dict[str, Any]]],
    ) -> None:
        """Process a single batch and update storage.

        Args:
            batch_data: Raw batch data from dataloader
            model: ComponentModel to get activations from
            tokenizer: Tokenizer for decoding
            device: Device for computation
            sigmoid_type: Sigmoid type for activation computation
            cluster_id_map: Mapping from cluster indices to ClusterId objects
            cluster_components: Component info for each cluster
        """
        # Extract and move batch to device
        batch: Int[Tensor, "batch_size n_ctx"] = extract_batch_data(batch_data).to(device)
        batch_size: int
        seq_len: int
        batch_size, seq_len = batch.shape

        # Get component activations from model
        activations: dict[str, Float[Tensor, "n_steps C"]] = component_activations(
            model,
            device,
            batch=batch,
            sigmoid_type=sigmoid_type,
        )
        processed: ProcessedActivations = process_activations(
            activations, seq_mode="concat", filter_dead_threshold=0
        )

        # Tokenize and create text samples
        batch_text_samples: list[TextSample] = _tokenize_and_create_text_samples(
            batch=batch,
            tokenizer=tokenizer,
            text_samples=self.text_samples,
        )

        # Compute cluster activations
        cluster_acts: ClusterActivations = compute_all_cluster_activations(
            processed=processed,
            cluster_components=cluster_components,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Store activations
        self._store_activations(
            cluster_acts=cluster_acts,
            processed=processed,
            batch_text_samples=batch_text_samples,
            batch_size=batch_size,
            seq_len=seq_len,
            cluster_id_map=cluster_id_map,
            cluster_components=cluster_components,
        )

    def _store_activations(
        self,
        cluster_acts: ClusterActivations,
        processed: ProcessedActivations,
        batch_text_samples: list[TextSample],
        batch_size: int,
        seq_len: int,
        cluster_id_map: dict[int, ClusterId],
        cluster_components: dict[int, list[dict[str, Any]]],
    ) -> None:
        """Store cluster-level and component-level activations from batch.

        Args:
            cluster_acts: Computed cluster activations
            processed: Processed component activations
            batch_text_samples: TextSample objects for the batch
            batch_size: Batch size
            seq_len: Sequence length
            cluster_id_map: Mapping from cluster indices to ClusterId objects
            cluster_components: Component info for each cluster
        """
        # Store for coactivation computation
        self.all_cluster_activations.append(cluster_acts)

        # Reshape to [batch_size, seq_len, n_clusters] for easier indexing
        acts_3d: Float[Tensor, "batch_size seq_len n_clusters"] = cluster_acts.activations.view(
            batch_size, seq_len, -1
        )
        acts_3d_cpu: Float[np.ndarray, "batch_size seq_len n_clusters"] = acts_3d.cpu().numpy()

        # Store activations per cluster
        for cluster_col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices):
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

                # Store cluster-level activations
                activations_np: Float[np.ndarray, " n_ctx"] = cluster_acts_2d[batch_sample_idx]
                self.cluster_activations[current_cluster_hash].append(activations_np)
                self.cluster_text_hashes[current_cluster_hash].append(text_hash)
                self.cluster_tokens[current_cluster_hash].append(text_sample.tokens)

                # Store component-level activations
                components_in_cluster: list[dict[str, Any]] = cluster_components[cluster_idx]
                for component_info in components_in_cluster:
                    component_label: str = component_info["label"]
                    comp_idx: int | None = processed.get_label_index(component_label)

                    if comp_idx is not None:
                        sample_offset: int = batch_sample_idx * seq_len
                        comp_acts_1d: Float[np.ndarray, " seq_len"] = (
                            processed.activations[sample_offset : sample_offset + seq_len, comp_idx]
                            .cpu()
                            .numpy()
                        )

                        self.component_activations[current_cluster_hash][component_label].append(
                            comp_acts_1d
                        )
                        self.component_text_hashes[current_cluster_hash][component_label].append(
                            text_hash
                        )


def _compute_cluster_activations(  # pyright: ignore[reportUnusedFunction]
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
    cluster_components: dict[int, list[dict[str, Any]]],
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
        components: list[dict[str, Any]] = cluster_components[cluster_idx]

        # Get component indices for this cluster
        comp_indices: list[int] = []
        for component_info in components:
            label: str = component_info["label"]
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


def compute_component_coactivations_in_cluster(
    processed: ProcessedActivations,
    component_labels: list[str],
    activation_threshold: float = 0.0,
) -> Float[np.ndarray, "n_comps n_comps"]:
    """Compute coactivation matrix for components within a cluster.

    Args:
        processed: ProcessedActivations containing all component activations
        component_labels: List of component labels in this cluster
        activation_threshold: Threshold for considering a component "active"

    Returns:
        Coactivation matrix where coact[i,j] is the number of samples where
        both component i and j activate above threshold
    """
    n_components: int = len(component_labels)

    # Get indices for these components
    comp_indices: list[int] = []
    for label in component_labels:
        comp_idx: int | None = processed.get_label_index(label)
        if comp_idx is not None:
            comp_indices.append(comp_idx)

    if not comp_indices:
        return np.zeros((n_components, n_components), dtype=np.float32)

    # Extract activations for these components: [n_samples, n_components]
    component_acts: Float[Tensor, "n_samples n_comps"] = processed.activations[:, comp_indices]

    # Binarize activations (1 if component activates above threshold, 0 otherwise)
    activation_mask: Float[Tensor, "n_samples n_comps"] = (
        component_acts > activation_threshold
    ).float()

    # Compute coactivation matrix: coact[i,j] = sum over samples of (comp_i_active * comp_j_active)
    coact: Float[Tensor, "n_comps n_comps"] = activation_mask.T @ activation_mask

    return coact.cpu().numpy()


def compute_component_cosine_similarities(
    processed: ProcessedActivations,
    component_labels: list[str],
) -> Float[np.ndarray, "n_comps n_comps"]:
    """Compute cosine similarity matrix for components within a cluster.

    Args:
        processed: ProcessedActivations containing all component activations
        component_labels: List of component labels in this cluster

    Returns:
        Cosine similarity matrix where sim[i,j] is the cosine similarity
        between component i and j activation patterns
    """
    n_components: int = len(component_labels)

    # Get indices for these components
    comp_indices: list[int] = []
    for label in component_labels:
        comp_idx: int | None = processed.get_label_index(label)
        if comp_idx is not None:
            comp_indices.append(comp_idx)

    if not comp_indices:
        return np.zeros((n_components, n_components), dtype=np.float32)

    # Extract activations for these components: [n_samples, n_components]
    component_acts: Float[Tensor, "n_samples n_comps"] = processed.activations[:, comp_indices]

    # Normalize each component's activation vector (columns)
    # L2 norm for each component across samples
    norms: Float[Tensor, " n_comps"] = torch.norm(component_acts, p=2, dim=0)

    # Avoid division by zero
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))

    # Normalize: [n_samples, n_comps]
    normalized_acts: Float[Tensor, "n_samples n_comps"] = component_acts / norms.unsqueeze(0)

    # Compute cosine similarity matrix: sim[i,j] = normalized_i · normalized_j
    cosine_sim: Float[Tensor, "n_comps n_comps"] = normalized_acts.T @ normalized_acts

    return cosine_sim.cpu().numpy()


@dataclass
class GlobalActivationsAccumulator:
    """Accumulates activations across all clusters for final DashboardData."""

    activations_map: dict[ActivationSampleHash, int]
    activations_list: list[Float[np.ndarray, " n_ctx"]]
    text_hashes_list: list[TextSampleHash]
    current_idx: int

    @classmethod
    def create(cls) -> "GlobalActivationsAccumulator":
        """Create empty accumulator."""
        return cls(
            activations_map={},
            activations_list=[],
            text_hashes_list=[],
            current_idx=0,
        )

    def add_cluster_data(
        self,
        cluster_data: ClusterData,
        activation_batch: ActivationSampleBatch,
        storage: BatchProcessingStorage,
    ) -> None:
        """Add cluster and component activations to global storage.

        Args:
            cluster_data: ClusterData with component-level data
            activation_batch: Activation batch for cluster-level data
            storage: Storage containing component activations
        """
        cluster_hash: ClusterIdHash = cluster_data.cluster_hash

        # Add cluster-level activations
        act_hashes: list[ActivationSampleHash] = activation_batch.activation_hashes
        for i, (text_hash, acts) in enumerate(
            zip(activation_batch.text_hashes, activation_batch.activations, strict=True)
        ):
            self.activations_map[act_hashes[i]] = self.current_idx
            self.activations_list.append(acts)
            self.text_hashes_list.append(text_hash)
            self.current_idx += 1

        # Add component-level activations
        if cluster_data.component_activations:
            for comp_label in cluster_data.component_activations:
                comp_acts_list = storage.component_activations[cluster_hash][comp_label]
                comp_text_hashes = storage.component_text_hashes[cluster_hash][comp_label]

                if not comp_acts_list:
                    continue

                comp_acts_array: Float[np.ndarray, "n_samples n_ctx"] = np.stack(comp_acts_list)

                for text_hash, comp_acts in zip(comp_text_hashes, comp_acts_array, strict=True):
                    comp_act_hash = ActivationSampleHash(f"{cluster_hash}:{comp_label}:{text_hash}")
                    self.activations_map[comp_act_hash] = self.current_idx
                    self.activations_list.append(comp_acts)
                    self.text_hashes_list.append(text_hash)
                    self.current_idx += 1


def _tokenize_and_create_text_samples(
    batch: Int[Tensor, "batch_size n_ctx"],
    tokenizer: PreTrainedTokenizer,
    text_samples: dict[TextSampleHash, TextSample],
) -> list[TextSample]:
    """Tokenize batch and create TextSample objects.

    Args:
        batch: Input token IDs
        tokenizer: Tokenizer for decoding
        text_samples: Existing text samples dict (for deduplication)

    Returns:
        List of TextSample objects for the batch
    """
    # Move to CPU and convert to list
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

    return batch_text_samples


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
            cluster_id_map=cluster_id_map,
            cluster_components=cluster_components,
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
            accumulator.add_cluster_data(cluster_data, activation_batch, storage)

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
