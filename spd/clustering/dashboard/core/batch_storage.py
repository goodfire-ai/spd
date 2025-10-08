"""Batch processing storage for accumulating activations."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from muutils.spinner import SpinnerContext
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.dashboard.core.base import (
    ClusterId,
    ClusterIdHash,
    TextSample,
    TextSampleHash,
)
from spd.clustering.dashboard.core.compute_helpers import (
    ClusterActivations,
    compute_all_cluster_activations,
)
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data

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

    with SpinnerContext(message="tokenizing: batch convert ids to tokens"):
        batch_token_strings: list[list[str]] = [
            tokenizer.convert_ids_to_tokens(seq)
            for seq in batch
        ]

    # Create text samples for entire batch
    batch_size: int = batch.shape[0]
    batch_text_samples: list[TextSample] = []
    for token_strings in tqdm(batch_token_strings, total=batch_size):
        text: str = " ".join(token_strings)
        text_sample: TextSample = TextSample(full_text=text, tokens=token_strings)
        text_samples[text_sample.text_hash] = text_sample
        batch_text_samples.append(text_sample)

    return batch_text_samples


@dataclass(slots=True, kw_only=True)
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
        cluster_id_map: Pre-computed mapping from cluster indices to ClusterIds
        cluster_components: Pre-computed component info for each cluster
    """

    cluster_activations: dict[ClusterIdHash, list[Float[np.ndarray, " n_ctx"]]]
    cluster_text_hashes: dict[ClusterIdHash, list[TextSampleHash]]
    cluster_tokens: dict[ClusterIdHash, list[list[str]]]
    component_activations: dict[ClusterIdHash, dict[str, list[Float[np.ndarray, " n_ctx"]]]]
    component_text_hashes: dict[ClusterIdHash, dict[str, list[TextSampleHash]]]
    text_samples: dict[TextSampleHash, TextSample]
    all_cluster_activations: list[ClusterActivations]
    # Pre-computed to avoid recomputation
    cluster_id_map: dict[int, ClusterId]
    cluster_components: dict[int, list[dict[str, Any]]]
    cluster_hash_map: dict[int, ClusterIdHash]

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

        # Compute cluster hash strings once to avoid redundant .to_string() calls
        cluster_hashes: dict[int, ClusterIdHash] = {
            idx: cluster_id_map[idx].to_string() for idx in unique_cluster_indices
        }

        cluster_activations: dict[ClusterIdHash, list[Float[np.ndarray, " n_ctx"]]] = {
            cluster_hashes[idx]: [] for idx in unique_cluster_indices
        }
        cluster_text_hashes: dict[ClusterIdHash, list[TextSampleHash]] = {
            cluster_hashes[idx]: [] for idx in unique_cluster_indices
        }
        cluster_tokens: dict[ClusterIdHash, list[list[str]]] = {
            cluster_hashes[idx]: [] for idx in unique_cluster_indices
        }
        component_activations: dict[ClusterIdHash, dict[str, list[Float[np.ndarray, " n_ctx"]]]] = {
            cluster_hashes[idx]: {comp["label"]: [] for comp in cluster_components[idx]}
            for idx in unique_cluster_indices
        }
        component_text_hashes: dict[ClusterIdHash, dict[str, list[TextSampleHash]]] = {
            cluster_hashes[idx]: {comp["label"]: [] for comp in cluster_components[idx]}
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
            cluster_id_map=cluster_id_map,
            cluster_components=cluster_components,
            cluster_hash_map=cluster_hashes,
        )

    def process_batch(
        self,
        batch_data: Any,
        model: ComponentModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        sigmoid_type: SigmoidTypes,
    ) -> None:
        """Process a single batch and update storage.

        Args:
            batch_data: Raw batch data from dataloader
            model: ComponentModel to get activations from
            tokenizer: Tokenizer for decoding
            device: Device for computation
            sigmoid_type: Sigmoid type for activation computation
        """
        # Extract and move batch to device
        batch: Int[Tensor, "batch_size n_ctx"] = extract_batch_data(batch_data).to(device)
        batch_size: int
        seq_len: int
        batch_size, seq_len = batch.shape

        with SpinnerContext(message="Computing component activations"):
            activations: dict[str, Float[Tensor, "n_steps C"]] = component_activations(
                model,
                device,
                batch=batch,
                sigmoid_type=sigmoid_type,
            )

        with SpinnerContext(message="Processing activations"):
            print("\n\nA1\n\n", flush=True)
            processed: ProcessedActivations = process_activations(
                activations, seq_mode="concat", filter_dead_threshold=0
            )
            print("\n\nA2\n\n", flush=True)

        print("\n\nA3\n\n", flush=True)
        batch_text_samples: list[TextSample] = _tokenize_and_create_text_samples(
            batch=batch,
            tokenizer=tokenizer,
            text_samples=self.text_samples,
        )

        with SpinnerContext(message="Computing cluster activations"):
            cluster_acts: ClusterActivations = compute_all_cluster_activations(
                processed=processed,
                cluster_components=self.cluster_components,
                batch_size=batch_size,
                seq_len=seq_len,
            )

        self._store_activations(
            cluster_acts=cluster_acts,
            processed=processed,
            batch_text_samples=batch_text_samples,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def _store_activations(
        self,
        cluster_acts: ClusterActivations,
        processed: ProcessedActivations,
        batch_text_samples: list[TextSample],
        batch_size: int,
        seq_len: int,
    ) -> None:
        """Store cluster-level and component-level activations from batch.

        Args:
            cluster_acts: Computed cluster activations
            processed: Processed component activations
            batch_text_samples: TextSample objects for the batch
            batch_size: Batch size
            seq_len: Sequence length
        """
        # Store for coactivation computation
        self.all_cluster_activations.append(cluster_acts)

        # Move all GPU→CPU transfers outside loops (CRITICAL OPTIMIZATION)
        # Reshape cluster activations to [batch_size, seq_len, n_clusters]
        acts_3d: Float[Tensor, "batch_size seq_len n_clusters"] = cluster_acts.activations.view(
            batch_size, seq_len, -1
        )
        acts_3d_cpu: Float[np.ndarray, "batch_size seq_len n_clusters"] = acts_3d.cpu().numpy()

        # Move component activations to CPU and reshape to [batch_size, seq_len, n_components]
        processed_acts_cpu: Float[np.ndarray, "batch*seq n_components"] = (
            processed.activations.cpu().numpy()
        )
        processed_acts_3d: Float[np.ndarray, "batch_size seq_len n_components"] = (
            processed_acts_cpu.reshape(batch_size, seq_len, -1)
        )

        # Pre-extract text hashes and tokens to avoid repeated lookups
        batch_text_hashes: list[TextSampleHash] = [
            sample.text_hash for sample in batch_text_samples
        ]
        batch_tokens: list[list[str]] = [sample.tokens for sample in batch_text_samples]

        # Pre-compute component label→index mapping to avoid repeated lookups
        # Build set of all component labels we'll need
        all_component_labels: set[str] = set()
        for cluster_components_list in self.cluster_components.values():
            for comp_info in cluster_components_list:
                all_component_labels.add(comp_info["label"])

        # Cache the indices
        label_to_index: dict[str, int | None] = {
            label: processed.get_label_index(label) for label in all_component_labels
        }

        # Filter empty clusters early - find which clusters have non-zero activations
        active_cluster_mask: Float[np.ndarray, " n_clusters"] = (
            np.abs(acts_3d_cpu).max(axis=(0, 1)) > 0
        )
        active_cluster_indices: list[tuple[int, int]] = [
            (col_idx, cluster_idx)
            for col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices)
            if active_cluster_mask[col_idx]
        ]

        # Store activations per cluster (only active clusters)
        for cluster_col_idx, cluster_idx in tqdm(
            active_cluster_indices,
            total=len(active_cluster_indices),
            desc="  Storing cluster activations",
            leave=True,
        ):
            cluster_acts_2d: Float[np.ndarray, "batch_size seq_len"] = acts_3d_cpu[
                :, :, cluster_col_idx
            ]

            # Use pre-computed hash instead of calling .to_string()
            current_cluster_hash: ClusterIdHash = self.cluster_hash_map[cluster_idx]

            # Get components for this cluster once (move invariant out of batch loop)
            components_in_cluster: list[dict[str, Any]] = self.cluster_components[cluster_idx]

            # Cache dictionary lookups - get list references once
            cluster_acts_list = self.cluster_activations[current_cluster_hash]
            cluster_text_hashes_list = self.cluster_text_hashes[current_cluster_hash]
            cluster_tokens_list = self.cluster_tokens[current_cluster_hash]

            # Vectorize storage - use .extend() instead of repeated .append()
            # Store cluster-level activations for entire batch at once
            cluster_acts_list.extend(cluster_acts_2d[i] for i in range(batch_size))
            cluster_text_hashes_list.extend(batch_text_hashes)
            cluster_tokens_list.extend(batch_tokens)

            # Store component-level activations
            for component_info in components_in_cluster:
                component_label: str = component_info["label"]
                comp_idx: int | None = label_to_index[component_label]

                if comp_idx is not None:
                    # Cache nested dictionary lookups
                    comp_acts_list = self.component_activations[current_cluster_hash][
                        component_label
                    ]
                    comp_text_hashes_list = self.component_text_hashes[current_cluster_hash][
                        component_label
                    ]

                    # Collect all component activations for this batch
                    batch_comp_acts = [processed_acts_3d[i, :, comp_idx] for i in range(batch_size)]

                    # Extend once instead of appending in loop
                    comp_acts_list.extend(batch_comp_acts)
                    comp_text_hashes_list.extend(batch_text_hashes)
