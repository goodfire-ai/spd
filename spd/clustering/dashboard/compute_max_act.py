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
    ActivationSample,
    ClusterActivationTracker,
    ClusterId,
    ClusterLabel,
    TextSample,
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
    dataset_index: int,
    tokenizer: PreTrainedTokenizer,
) -> tuple[TextSample, ActivationSample]:
    """Create TextSample and ActivationSample from batch data.

    Args:
        cluster_id: ClusterId this sample belongs to
        batch: Input token batch
        batch_idx: Index within batch
        sequence_acts: Activations for entire sequence
        dataset_index: Index in original dataset
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

    # Create TextSample (without activations)
    text_sample = TextSample(
        full_text=text,
        tokens=token_strings,
        dataset_index=dataset_index,
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

    # Get unique cluster indices and component info using MergeHistory methods
    unique_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid)
        for cid in unique_cluster_indices
    }

    # Create ClusterId objects for each cluster
    # TODO: Get actual spd_run, clustering_run from context
    cluster_ids: list[ClusterId] = [
        ClusterId(
            spd_run="unknown",  # TODO: populate from wandb run
            clustering_run="unknown",  # TODO: populate from wandb run
            iteration=iteration,
            cluster_label=ClusterLabel(
                module_name="unknown",  # TODO: extract from merge_history
                original_index=idx,
            ),
            __DELETEME__=idx,
        )
        for idx in unique_cluster_indices
    ]

    # Create mapping from cluster_index to ClusterId for easy lookup
    cluster_id_map: dict[int, ClusterId] = {cid.__DELETEME__: cid for cid in cluster_ids}

    # Define tracking criteria (default: just max_activation)
    from spd.clustering.dashboard.text_sample import TrackingCriterion

    criteria = [
        TrackingCriterion(
            property_name="max_activation",
            direction="max",
            n_samples=n_samples,
        )
    ]

    # Initialize tracker
    tracker = ClusterActivationTracker(
        cluster_ids=cluster_ids,
        criteria=criteria,
        device=device,
    )

    # Build text pool separately
    from spd.clustering.dashboard.text_sample import TextHash

    text_pool: dict[TextHash, TextSample] = {}
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

        for cluster_idx in unique_cluster_indices:
            # Compute cluster activations
            acts_2d: Float[Tensor, "batch_size seq_len"] = _compute_cluster_activations(
                processed, cluster_components[cluster_idx], batch_size, seq_len
            )

            if acts_2d.abs().max() == 0:
                continue

            # Find top activations across all positions
            flat_acts: Float[Tensor, " batch_size*seq_len"] = acts_2d.flatten()
            k: int = min(n_samples, len(flat_acts))
            top_vals: Float[Tensor, " k"]
            top_idx: Int[Tensor, " k"]
            top_vals, top_idx = torch.topk(flat_acts, k)

            # Get ClusterId for this cluster
            cluster_id: ClusterId = cluster_id_map[cluster_idx]

            # Create samples for batch insertion
            activation_samples: list[ActivationSample] = []
            for idx in top_idx:
                batch_idx_i: int = int(idx // seq_len)
                current_dataset_idx: int = dataset_idx_counter + batch_idx_i

                text_sample, act_sample = _create_samples(
                    cluster_id=cluster_id,
                    batch=batch,
                    batch_idx=batch_idx_i,
                    sequence_acts=acts_2d[batch_idx_i],
                    dataset_index=current_dataset_idx,
                    tokenizer=tokenizer,
                )

                # Add text sample to external pool
                text_pool[text_sample.text_hash] = text_sample
                activation_samples.append(act_sample)

            # Batch insert activation samples into tracker
            tracker.try_insert_batch(activation_samples)

        dataset_idx_counter += batch_size

    return tracker.to_result_dict(cluster_components, text_pool)
