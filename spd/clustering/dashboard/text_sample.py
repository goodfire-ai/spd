"""Data structures for tracking max-activating text samples."""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor


# TODO: this should also contain some kind of key, telling us *which cluster* the activations correspond to
@dataclass
class TextSample:
    """A text sample with activation information."""

    full_text: str  # Original full context
    dataset_index: int  # Index from original dataset
    tokens: list[str]  # Token strings
    activations: list[float]  # Activation for each token
    # TODO: the below should be *properties* that compute on demand
    mean_activation: float
    median_activation: float
    max_activation: float
    max_position: int  # Position of max activation

    def serialize(self) -> dict[str, Any]:
        """Serialize the TextSample to a dictionary."""
        return asdict(self)


# TODO: given some TextSample, rather than keeping track of text with the highest `max(activations)`, we might want to keep track of text which maximizes or minimizes some other criterion, which we can assume to be a property on `TextSample`. rewrite this as `ClusterActivationTracker` which takes cluster ids, n_samples, device, and a list of... some kind of "TextSampleStat" named tuple thing, which tells us which property to use for ranking, and whether we care about max or min. adjust outputs accordingly. Also, maybe we want a different strategy for actually *storing* the data -- rather than storing many copies of the text/activations, we just store a dataset index, and then filter the original dataset at the end? we also would like to decouple the text from the activations, since the same text might be max/min activating for multiple different clusters
class ClusterMaxTracker:
    """Tracks top-k max-activating samples per cluster."""

    def __init__(self, cluster_ids: list[int], n_samples: int, device: torch.device):
        self.n_samples: int = n_samples
        self.device: torch.device = device

        # Initialize tracking structures
        self.max_acts: dict[int, Float[Tensor, " n_samples"]] = {
            cid: torch.full((n_samples,), -1e10, device=device) for cid in cluster_ids
        }
        self.max_texts: dict[int, list[TextSample | None]] = {
            cid: [None] * n_samples for cid in cluster_ids
        }
        self.used_dataset_indices: dict[int, set[int]] = {cid: set() for cid in cluster_ids}

    def try_insert_batch(
        self,
        cluster_id: int,
        vals: Float[Tensor, " k"],
        text_samples: list[TextSample],
    ) -> int:
        """Try to insert multiple text samples if they're in the top-k for the cluster.

        Args:
            cluster_id: Cluster ID
            vals: Activation values (length k)
            text_samples: TextSamples to insert (length k)

        Returns:
            Number of samples successfully inserted
        """
        assert len(vals) == len(text_samples), "vals and text_samples must have same length"

        n_inserted: int = 0
        for val, text_sample in zip(vals, text_samples, strict=True):
            # Skip if we've already used this dataset index for this cluster
            if text_sample.dataset_index in self.used_dataset_indices[cluster_id]:
                continue

            # Find insertion point
            for j in range(self.n_samples):
                if val > self.max_acts[cluster_id][j]:
                    # Shift and insert
                    if j < self.n_samples - 1:
                        self.max_acts[cluster_id][j + 1 :] = self.max_acts[cluster_id][j:-1].clone()
                        self.max_texts[cluster_id][j + 1 :] = self.max_texts[cluster_id][j:-1]

                    self.max_acts[cluster_id][j] = val
                    self.max_texts[cluster_id][j] = text_sample
                    self.used_dataset_indices[cluster_id].add(text_sample.dataset_index)
                    n_inserted += 1
                    break

        return n_inserted

    def to_result_dict(
        self, cluster_components: dict[int, list[dict[str, Any]]]
    ) -> dict[int, dict[str, list[dict[str, Any]]]]:
        """Convert tracking state to final result dictionary.

        Args:
            cluster_components: Mapping from cluster_id to component info dicts

        Returns:
            Dict mapping cluster_id to dict with keys "components" and "samples"
        """
        result: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for cluster_id in self.max_texts:
            samples: list[TextSample] = [s for s in self.max_texts[cluster_id] if s is not None]
            result[cluster_id] = {
                "components": cluster_components[cluster_id],
                "samples": [s.serialize() for s in samples],
            }
        return result
