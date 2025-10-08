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
    # Move to CPU
    batch_cpu: Int[Tensor, "batch_size n_ctx"] = batch.cpu()
    batch_size, n_ctx = batch_cpu.shape

    # Batch decode full texts
    batch_texts: list[str] = tokenizer.batch_decode(batch_cpu)  # pyright: ignore[reportAttributeAccessIssue]

    # Batch decode individual tokens - reshape and pass tensor directly
    # Reshape [batch_size, n_ctx] -> [batch_size * n_ctx, 1]
    flattened_tokens: Int[Tensor, "total 1"] = batch_cpu.reshape(-1, 1)
    all_token_strings: list[str] = tokenizer.batch_decode(flattened_tokens)  # pyright: ignore[reportAttributeAccessIssue]

    # Reshape back to [batch_size, n_ctx]
    batch_token_strings: list[list[str]] = [
        all_token_strings[i * n_ctx : (i + 1) * n_ctx] for i in range(batch_size)
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
            processed: ProcessedActivations = process_activations(
                activations, seq_mode="concat", filter_dead_threshold=0
            )

        with SpinnerContext(message="tokenizing and creating text samples"):
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

        # Reshape to [batch_size, seq_len, n_clusters] for easier indexing
        acts_3d: Float[Tensor, "batch_size seq_len n_clusters"] = cluster_acts.activations.view(
            batch_size, seq_len, -1
        )
        acts_3d_cpu: Float[np.ndarray, "batch_size seq_len n_clusters"] = acts_3d.cpu().numpy()

        # Store activations per cluster
        for cluster_col_idx, cluster_idx in tqdm(
            enumerate(cluster_acts.cluster_indices),
            total=len(cluster_acts.cluster_indices),
            desc="  Storing cluster activations",
            leave=False,
        ):
            cluster_acts_2d: Float[np.ndarray, "batch_size seq_len"] = acts_3d_cpu[
                :, :, cluster_col_idx
            ]

            # Skip if no activations
            if np.abs(cluster_acts_2d).max() == 0:
                continue

            current_cluster_id: ClusterId = self.cluster_id_map[cluster_idx]
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
                components_in_cluster: list[dict[str, Any]] = self.cluster_components[cluster_idx]
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
