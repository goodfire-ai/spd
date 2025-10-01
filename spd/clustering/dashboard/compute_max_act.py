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
from spd.clustering.dashboard.text_sample import (
    ActivationSample,
    ClusterCriterionTracker,
    ClusterId,
    ClusterIdHash,
    ClusterLabel,
    TextSample,
    TextSampleHash,
    TrackingCriterion,
    TrackingCriterionHash,
)
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data, get_module_device


def _create_samples(
    cluster_id: ClusterId,
    batch: Int[Tensor, "batch_size n_ctx"],
    batch_idx: int,
    sequence_acts: Float[Tensor, " seq_len"],
    tokenizer: PreTrainedTokenizer,
) -> tuple[TextSample, ActivationSample]:
    """Create TextSample and ActivationSample from batch data.

    Args:
        cluster_id: ClusterId this sample belongs to
        batch: Input token batch
        batch_idx: Index within batch
        sequence_acts: Activations for entire sequence
        tokenizer: Tokenizer for decoding

    Returns:
        Tuple of (TextSample, ActivationSample)
    """
    # Extract full sequence data
    tokens: Int[Tensor, " n_ctx"] = batch[batch_idx].cpu()
    tokens_list: list[int] = tokens.tolist()
    text: str = tokenizer.decode(tokens)  # pyright: ignore[reportAttributeAccessIssue]

    # Convert token IDs to token strings
    token_strings: list[str] = [
        tokenizer.decode([tid])  # pyright: ignore[reportAttributeAccessIssue]
        for tid in tokens_list
    ]

    # Create TextSample
    text_sample = TextSample(
        full_text=text,
        tokens=token_strings,
    )

    # Create ActivationSample (with numpy array)
    activations_np: np.ndarray = sequence_acts.cpu().numpy()
    activation_sample = ActivationSample(
        cluster_id=cluster_id,
        text_hash=text_sample.text_hash,
        activations=activations_np,
    )

    return text_sample, activation_sample


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
) -> dict[int, dict[str, Any]]:
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

    Returns:
        Dict mapping cluster_id to results with components and samples per criterion
    """
    device: torch.device = get_module_device(model)

    # Get unique cluster indices and component info
    unique_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid)
        for cid in unique_cluster_indices
    }

    # Create ClusterId objects for each cluster
    cluster_ids: list[ClusterId] = []
    for idx in unique_cluster_indices:
        # Extract cluster label from first component
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

        cluster_ids.append(
            ClusterId(
                spd_run="unknown",  # TODO: populate from wandb run
                clustering_run="unknown",  # TODO: populate from wandb run
                iteration=iteration,
                cluster_label=cluster_label,
            )
        )

    # Create mapping from cluster_index to ClusterId
    cluster_id_map: dict[int, ClusterId] = {
        unique_cluster_indices[i]: cluster_ids[i] for i in range(len(cluster_ids))
    }

    # Define tracking criteria
    criteria = [
        TrackingCriterion(
            property_name="max_activation",
            direction="max",
            n_samples=n_samples,
        )
    ]

    # Initialize tracker and pools
    tracker = ClusterCriterionTracker.create_empty()
    text_pool: dict[TextSampleHash, TextSample] = {}
    activation_pool: dict[tuple[TextSampleHash, ClusterIdHash], ActivationSample] = {}

    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches)):
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

            # Process all samples in this batch for this cluster
            cluster_id: ClusterId = cluster_id_map[cluster_idx]
            cluster_hash: ClusterIdHash = cluster_id.to_string()

            for batch_idx_i in range(batch_size):
                text_sample, act_sample = _create_samples(
                    cluster_id=cluster_id,
                    batch=batch,
                    batch_idx=batch_idx_i,
                    sequence_acts=acts_2d[batch_idx_i],
                    tokenizer=tokenizer,
                )

                # Add to pools
                text_hash = text_sample.text_hash
                text_pool[text_hash] = text_sample
                activation_pool[(text_hash, cluster_hash)] = act_sample

                # Compute stat values for all criteria
                stat_values: dict[TrackingCriterionHash, float] = {}
                for criterion in criteria:
                    stat_value = getattr(act_sample, criterion.property_name)
                    stat_values[criterion.to_string()] = stat_value

                # Add to tracker
                tracker.add_sample(cluster_hash, text_hash, stat_values)

    # Filter to top-k
    top_samples = tracker.filter_top_k(criteria)

    # Build result dict
    result: dict[int, dict[str, Any]] = {}
    for i, cluster_idx in enumerate(unique_cluster_indices):
        cluster_id = cluster_ids[i]
        cluster_hash = cluster_id.to_string()

        cluster_result: dict[str, Any] = {
            "components": cluster_components[cluster_idx],
            "samples": {},
        }

        if cluster_hash in top_samples:
            for crit_hash, text_hashes in top_samples[cluster_hash].items():
                # Convert text_hashes to full sample dicts
                samples_list = []
                for text_hash in text_hashes:
                    text_sample = text_pool[text_hash]
                    act_sample = activation_pool[(text_hash, cluster_hash)]
                    samples_list.append({
                        "text": text_sample.full_text,
                        "tokens": text_sample.tokens,
                        "activations": act_sample.activations.tolist(),
                        "max_activation": act_sample.max_activation,
                        "max_position": act_sample.max_position,
                    })
                cluster_result["samples"][crit_hash] = samples_list

        result[cluster_idx] = cluster_result

    return result




"""
 #######  ##       ########
##     ## ##       ##     ##
##     ## ##       ##     ##
##     ## ##       ##     ##
##     ## ##       ##     ##
##     ## ##       ##     ##
 #######  ######## ########
"""




@dataclass
class ClusterActivationDatabase:
    """Database for storing text samples and activations per cluster.

    Database tables:
    - text_samples: {TextHash: TextSample}
    - cluster_top_samples: {ClusterHash: {CriterionHash: [TextHash]}}
    - activation_samples: {(TextHash, ClusterHash): ActivationSample}
    """

    # Table: TextHash -> TextSample
    text_samples: dict[TextSampleHash, TextSample]

    # Table: ClusterHash -> {CriterionHash -> [TextHash]}
    cluster_top_samples: dict[ClusterIdHash, dict[TrackingCriterionHash, list[TextSampleHash]]]

    # Table: (TextHash, ClusterHash) -> ActivationSample
    activation_samples: dict[tuple[TextSampleHash, ClusterIdHash], ActivationSample]

    @classmethod
    def from_tracker(
        cls,
        tracker: ClusterCriterionTracker,
        text_samples: dict[TextSampleHash, TextSample],
        activation_samples: dict[tuple[TextSampleHash, ClusterIdHash], ActivationSample],
    ) -> "ClusterActivationDatabase":
        """Create database from a tracker and sample tables.

        Args:
            tracker: ClusterTopSamplesTracker with all clusters
            text_samples: Complete text samples table
            activation_samples: Complete activation samples table

        Returns:
            Complete ClusterActivationDatabase
        """
        cluster_top_samples = tracker.get_cluster_top_samples()

        return cls(
            text_samples=text_samples,
            cluster_top_samples=cluster_top_samples,
            activation_samples=activation_samples,
        )

    @classmethod
    def generate(
        cls,
        model: ComponentModel,
        sigmoid_type: SigmoidTypes,
        tokenizer: PreTrainedTokenizer,
        dataloader: DataLoader[Any],
        merge_history: MergeHistory,
        iteration: int,
        criteria: list[TrackingCriterion],
        spd_run: str,
        clustering_run: str,
        n_batches: int,
    ) -> "ClusterActivationDatabase":
        """Generate database by computing activations from model and data.

        Args:
            model: ComponentModel to get activations from
            sigmoid_type: Sigmoid type for activation computation
            tokenizer: Tokenizer for decoding text
            dataloader: DataLoader providing batches
            merge_history: MergeHistory containing cluster information
            iteration: Merge iteration to analyze
            criteria: List of tracking criteria
            spd_run: SPD run identifier
            clustering_run: Clustering run identifier
            n_batches: Number of batches to process

        Returns:
            Populated ClusterActivationDatabase
        """
        device: torch.device = get_module_device(model)

        # Get unique cluster indices and component info
        unique_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
        cluster_components: dict[int, list[dict[str, Any]]] = {
            cid: merge_history.get_cluster_components_info(iteration, cid)
            for cid in unique_cluster_indices
        }

        # Create ClusterId objects for each cluster
        cluster_ids: list[ClusterId] = []
        for idx in unique_cluster_indices:
            # Get cluster label from first component
            components = cluster_components[idx]
            if components:
                first_label = components[0]["label"]
                module_name, original_index_str = first_label.rsplit(":", 1)
                cluster_label = ClusterLabel(
                    module_name=module_name, original_index=int(original_index_str)
                )
            else:
                cluster_label = ClusterLabel(module_name="unknown", original_index=idx)

            cluster_ids.append(
                ClusterId(
                    spd_run=spd_run,
                    clustering_run=clustering_run,
                    iteration=iteration,
                    cluster_label=cluster_label,
                )
            )

        # Create mapping from cluster_index to ClusterId
        cluster_id_map: dict[int, ClusterId] = {
            unique_cluster_indices[i]: cluster_ids[i] for i in range(len(cluster_ids))
        }

        # Initialize empty database
        db = cls.create_empty(cluster_ids=cluster_ids, criteria=criteria)

        # Process batches
        for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches)):
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

                # Find top activations across all positions
                flat_acts: Float[Tensor, " batch_size*seq_len"] = acts_2d.flatten()
                # Get top k per criterion (max of n_samples across all criteria)
                k: int = min(max(c.n_samples for c in criteria), len(flat_acts))
                top_idx: Int[Tensor, " k"]
                _, top_idx = torch.topk(flat_acts, k)

                # Get ClusterId for this cluster
                cluster_id: ClusterId = cluster_id_map[cluster_idx]

                # Create and insert samples
                for idx in top_idx:
                    batch_idx_i: int = int(idx // seq_len)

                    text_sample, act_sample = _create_samples(
                        cluster_id=cluster_id,
                        batch=batch,
                        batch_idx=batch_idx_i,
                        sequence_acts=acts_2d[batch_idx_i],
                        tokenizer=tokenizer,
                    )

                    # Try to insert into database
                    db._try_insert_activation(act_sample, text_sample)

        return db


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


def _create_samples(
    cluster_id: ClusterId,
    batch: Int[Tensor, "batch_size n_ctx"],
    batch_idx: int,
    sequence_acts: Float[Tensor, " seq_len"],
    tokenizer: PreTrainedTokenizer,
) -> tuple[TextSample, ActivationSample]:
    """Create TextSample and ActivationSample from batch data.

    Args:
        cluster_id: ClusterId this sample belongs to
        batch: Input token batch
        batch_idx: Index within batch
        sequence_acts: Activations for entire sequence
        tokenizer: Tokenizer for decoding

    Returns:
        Tuple of (TextSample, ActivationSample)
    """
    # Extract full sequence data
    tokens: Int[Tensor, " n_ctx"] = batch[batch_idx].cpu()
    tokens_list: list[int] = tokens.tolist()
    text: str = tokenizer.decode(tokens)  # pyright: ignore[reportAttributeAccessIssue]

    # Convert token IDs to token strings
    token_strings: list[str] = [
        tokenizer.decode([tid])  # pyright: ignore[reportAttributeAccessIssue]
        for tid in tokens_list
    ]

    # Create TextSample
    text_sample = TextSample(
        full_text=text,
        tokens=token_strings,
    )

    # Create ActivationSample (with numpy array)
    activations_np: np.ndarray = sequence_acts.cpu().numpy()
    activation_sample = ActivationSample(
        cluster_id=cluster_id,
        text_hash=text_sample.text_hash,
        activations=activations_np,
    )

    return text_sample, activation_sample
