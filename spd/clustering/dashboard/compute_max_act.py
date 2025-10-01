"""Core computation logic for finding max-activating text samples."""

from typing import Any

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
from spd.clustering.dashboard.text_sample import ClusterMaxTracker, TextSample
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data, get_module_device


def _create_text_sample(
    batch: Int[Tensor, "batch_size n_ctx"],
    batch_idx: int,
    pos_idx: int,
    sequence_acts: Float[Tensor, " seq_len"],
    val: float,
    dataset_index: int,
    tokenizer: PreTrainedTokenizer,
) -> TextSample:
    """Create a TextSample from batch data and activations.

    Args:
        batch: Input token batch
        batch_idx: Index within batch
        pos_idx: Position of max activation
        sequence_acts: Activations for entire sequence
        val: Max activation value
        dataset_index: Index in original dataset
        tokenizer: Tokenizer for decoding

    Returns:
        TextSample instance
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

    # Get all activations for this sequence
    activations_list: list[float] = sequence_acts.cpu().tolist()

    # Compute statistics
    mean_act: float = float(sequence_acts.mean().item())
    median_act: float = float(sequence_acts.median().item())
    max_act: float = float(val)

    return TextSample(
        full_text=text,
        dataset_index=dataset_index,
        tokens=token_strings,
        activations=activations_list,
        mean_activation=mean_act,
        median_activation=median_act,
        max_activation=max_act,
        max_position=pos_idx,
    )


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
) -> dict[int, dict[str, list[dict[str, Any]]]]:
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
        Dict mapping cluster_id to dict with "components" and "samples" keys
    """
    device: torch.device = get_module_device(model)

    # Get unique clusters and component info using MergeHistory methods
    unique_clusters: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid) for cid in unique_clusters
    }

    # Initialize tracker
    tracker: ClusterMaxTracker = ClusterMaxTracker(unique_clusters, n_samples, device)
    dataset_idx_counter: int = 0

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

        for cluster_id in unique_clusters:
            # Compute cluster activations
            acts_2d: Float[Tensor, "batch_size seq_len"] = _compute_cluster_activations(
                processed, cluster_components[cluster_id], batch_size, seq_len
            )

            if acts_2d.abs().max() == 0:
                continue

            # Find top activations across all positions
            flat_acts: Float[Tensor, " batch_size*seq_len"] = acts_2d.flatten()
            k: int = min(n_samples, len(flat_acts))
            top_vals: Float[Tensor, " k"]
            top_idx: Int[Tensor, " k"]
            top_vals, top_idx = torch.topk(flat_acts, k)

            # Create TextSamples for batch insertion
            text_samples: list[TextSample] = []
            for val, idx in zip(top_vals, top_idx, strict=False):
                batch_idx_i: int = int(idx // seq_len)
                pos_idx: int = int(idx % seq_len)
                current_dataset_idx: int = dataset_idx_counter + batch_idx_i

                text_sample: TextSample = _create_text_sample(
                    batch=batch,
                    batch_idx=batch_idx_i,
                    pos_idx=pos_idx,
                    sequence_acts=acts_2d[batch_idx_i],
                    val=float(val),
                    dataset_index=current_dataset_idx,
                    tokenizer=tokenizer,
                )
                text_samples.append(text_sample)

            # Batch insert into tracker
            tracker.try_insert_batch(cluster_id, top_vals, text_samples)

        dataset_idx_counter += batch_size

    return tracker.to_result_dict(cluster_components)
