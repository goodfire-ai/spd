"""Data structures for tracking max-activating text samples."""

import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, NewType

import numpy as np
import torch

# Type aliases
TextHash = NewType("TextHash", str)
ActivationHash = NewType("ActivationHash", str)
ClusterHash = NewType("ClusterHash", str)
CriterionIndex = NewType("CriterionIndex", int)
Direction = Literal["max", "min"]

_SEPARATOR_1: str = "-"
_SEPARATOR_2: str = ":"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterLabel:
    """Label for a cluster component."""

    module_name: str
    original_index: int

    def __str__(self) -> str:
        """Format as 'module_name:original_index'."""
        return f"{self.module_name}:{self.original_index}"

    def as_tuple(self) -> tuple[str, int]:
        """Return as a tuple (module_name, original_index)."""
        return (self.module_name, self.original_index)

    def string(self) -> str:
        """Return as a string 'module_name:original_index'."""
        return f"{self.module_name}-{self.original_index}"

    def from_string(s: str) -> "ClusterLabel":
        """Create ClusterLabel from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_1)
        if len(parts) != 2:
            raise ValueError(f"Invalid ClusterLabel string: {s}")
        return ClusterLabel(module_name=parts[0], original_index=int(parts[1]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterId:
    """Unique identifier for a cluster. This should uniquely identify a cluster *globally*"""

    spd_run: str  # SPD run identifier
    clustering_run: str  # Clustering run identifier
    iteration: int  # Merge iteration number
    cluster_label: ClusterLabel  # Component label

    def to_list(self) -> list[Any]:
        """Return as a list of identifying components."""
        return [
            self.spd_run,
            self.clustering_run,
            self.iteration,
            self.cluster_label.module_name,
            self.cluster_label.original_index,
        ]

    def to_string(self) -> ClusterHash:
        """Hash uniquely identifying this cluster."""
        # Use all identifying information to create unique hash
        transformed_lst: list[str] = [str(part) for part in self.to_list()]
        assert all(_SEPARATOR_1 not in part for part in transformed_lst), (
            "Parts cannot contain separator"
        )
        assert all(_SEPARATOR_2 not in part for part in transformed_lst), (
            "Parts cannot contain separator"
        )

        return ClusterHash(_SEPARATOR_1.join(transformed_lst))

    @classmethod
    def from_string(cls, s: str) -> "ClusterId":
        """Create ClusterId from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_1)
        if len(parts) != 5:
            raise ValueError(f"Invalid ClusterId string: {s}")
        return cls(
            spd_run=parts[0],
            clustering_run=parts[1],
            iteration=int(parts[2]),
            cluster_label=ClusterLabel(module_name=parts[3], original_index=int(parts[4])),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TextSample:
    """Text content, with a reference to the dataset. depends on tokenizer used."""

    full_text: str
    tokens: list[str]

    @cached_property
    def text_hash(self) -> TextHash:
        """Hash of full_text for deduplication."""
        return TextHash(hashlib.sha256(self.full_text.encode()).hexdigest()[:8])

    def length(self) -> int:
        """Return the number of tokens."""
        return len(self.tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationSample:
    """Activations for a text sample in a specific cluster."""

    cluster_id: ClusterId
    text_hash: TextHash  # Reference to TextSample
    activations: np.ndarray  # Shape: (seq_len,)

    @cached_property
    def activation_hash(self) -> ActivationHash:
        """Hash uniquely identifying this activation sample (cluster_hash + text_hash)."""
        # Combine cluster_hash with text_hash to create unique identifier
        return ActivationHash(f"{self.cluster_id.to_string}:{self.text_hash}")

    @property
    def mean_activation(self) -> float:
        """Compute mean activation across all tokens."""
        return float(np.mean(self.activations))

    @property
    def min_activation(self) -> float:
        """Compute minimum activation across all tokens."""
        return float(np.min(self.activations))

    @property
    def median_activation(self) -> float:
        """Compute median activation across all tokens."""
        return float(np.median(self.activations))

    @property
    def max_activation(self) -> float:
        """Compute maximum activation across all tokens."""
        return float(np.max(self.activations))

    @property
    def max_position(self) -> int:
        """Find position of maximum activation."""
        return int(np.argmax(self.activations))

    def length(self) -> int:
        """Return the number of activation values (sequence length)."""
        return len(self.activations)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingCriterion:
    """Defines what statistics to track."""

    property_name: str
    "max_activation, mean_activation, etc. - must be a property on ActivationSample"

    direction: Direction

    n_samples: int


class ClusterActivationTracker:
    """Tracks top-k samples per cluster across multiple criteria.

    Does not store text data - only activation samples with activation_hash references.
    Text samples must be provided separately when serializing results.

    Internal structure:
    - activation_samples: {ActivationHash: ActivationSample}
    - top_samples: {CriterionIndex: {ClusterHash: [ActivationHash]}}
    """

    def __init__(
        self,
        cluster_ids: list[ClusterId],
        criteria: list[TrackingCriterion],
        device: torch.device,
    ):
        self.cluster_ids: list[ClusterId] = cluster_ids
        self.criteria: list[TrackingCriterion] = criteria
        self.device: torch.device = device

        # Store all activation samples by their unique hash
        self.activation_samples: dict[ActivationHash, ActivationSample] = {}

        # For each criterion, track top-k activation_hashes per cluster
        self.top_samples: dict[CriterionIndex, dict[ClusterHash, list[ActivationHash]]] = {}
        for crit_idx, _ in enumerate(criteria):
            crit_idx_typed = CriterionIndex(crit_idx)
            self.top_samples[crit_idx_typed] = {}
            for cluster_id in cluster_ids:
                self.top_samples[crit_idx_typed][cluster_id.to_string] = []

        # Track used text_hashes per cluster to avoid duplicates
        self.used_text_hashes: dict[ClusterHash, set[TextHash]] = {
            cid.to_string: set() for cid in cluster_ids
        }

    @property
    def required_text_hashes(self) -> set[TextHash]:
        """Compute the set of all text hashes needed by stored activation samples."""
        hashes: set[TextHash] = set()
        for act_sample in self.activation_samples.values():
            hashes.add(act_sample.text_hash)
        return hashes

    def try_insert_batch(
        self,
        activation_samples: list[ActivationSample],
    ) -> dict[CriterionIndex, int]:
        """Try to insert activation samples for all criteria.

        Args:
            activation_samples: ActivationSample objects to potentially insert

        Returns:
            Dict mapping CriterionIndex to number of samples inserted
        """
        n_inserted: dict[CriterionIndex, int] = {
            CriterionIndex(i): 0 for i in range(len(self.criteria))
        }

        for act_sample in activation_samples:
            # Get cluster_hash from the ClusterId
            cluster_hash: ClusterHash = act_sample.cluster_id.to_string

            # Skip if we've already used this text hash for this cluster
            if act_sample.text_hash in self.used_text_hashes[cluster_hash]:
                continue

            # Store activation sample
            act_hash: ActivationHash = act_sample.activation_hash
            self.activation_samples[act_hash] = act_sample

            # Try to insert for each criterion
            for crit_idx, criterion in enumerate(self.criteria):
                crit_idx_typed = CriterionIndex(crit_idx)
                # Get the value for this criterion
                stat_value: float = getattr(act_sample, criterion.property_name)

                top_list: list[ActivationHash] = self.top_samples[crit_idx_typed][cluster_hash]

                # Find insertion point based on direction
                insert_pos: int | None = None

                for j, existing_hash in enumerate(top_list):
                    existing_sample = self.activation_samples[existing_hash]
                    existing_value: float = getattr(existing_sample, criterion.property_name)

                    if (criterion.direction == "max" and stat_value > existing_value) or (
                        criterion.direction == "min" and stat_value < existing_value
                    ):
                        insert_pos = j
                        break

                # Insert if we found a position or list not full
                if insert_pos is not None:
                    top_list.insert(insert_pos, act_hash)
                    # Keep only top n_samples
                    if len(top_list) > criterion.n_samples:
                        top_list.pop()
                    n_inserted[crit_idx_typed] += 1
                elif len(top_list) < criterion.n_samples:
                    top_list.append(act_hash)
                    n_inserted[crit_idx_typed] += 1

            # Mark text hash as used after trying all criteria
            self.used_text_hashes[cluster_hash].add(act_sample.text_hash)

        return n_inserted

    def to_result_dict(
        self,
        cluster_components: dict[int, list[dict[str, Any]]],
        text_pool: dict[TextHash, TextSample],
    ) -> dict[int, dict[str, Any]]:
        """Convert tracking state to final result dictionary.

        Args:
            cluster_components: Mapping from cluster_index to component info dicts
            text_pool: External text pool mapping text_hash to TextSample

        Returns:
            Dict mapping cluster_index to results with components and samples per criterion
        """
        result: dict[int, dict[str, Any]] = {}

        for cluster_id in self.cluster_ids:
            cluster_idx: int = cluster_id.__DELETEME__
            cluster_hash: ClusterHash = cluster_id.to_string
            cluster_result: dict[str, Any] = {
                "cluster_id": {
                    "spd_run": cluster_id.spd_run,
                    "clustering_run": cluster_id.clustering_run,
                    "iteration": cluster_id.iteration,
                    "cluster_label": {
                        "module_name": cluster_id.cluster_label.module_name,
                        "original_index": cluster_id.cluster_label.original_index,
                        "str": str(cluster_id.cluster_label),
                    },
                    "cluster_index": cluster_id.__DELETEME__,
                },
                "components": cluster_components[cluster_idx],
                "criteria": {},
            }

            # For each criterion, collect samples
            for crit_idx, criterion in enumerate(self.criteria):
                crit_idx_typed = CriterionIndex(crit_idx)
                activation_hashes: list[ActivationHash] = self.top_samples[crit_idx_typed][
                    cluster_hash
                ]

                # Serialize samples with text data from external pool
                serialized: list[dict[str, Any]] = []
                for act_hash in activation_hashes:
                    act_sample = self.activation_samples[act_hash]
                    text_sample = text_pool[act_sample.text_hash]
                    sample_dict: dict[str, Any] = {
                        "full_text": text_sample.full_text,
                        "tokens": text_sample.tokens,
                        "dataset_index": text_sample.dataset_index,
                        "activations": act_sample.activations.tolist(),
                        "mean_activation": act_sample.mean_activation,
                        "median_activation": act_sample.median_activation,
                        "max_activation": act_sample.max_activation,
                        "min_activation": act_sample.min_activation,
                        "max_position": act_sample.max_position,
                    }
                    serialized.append(sample_dict)

                criterion_key: str = f"{criterion.property_name}_{criterion.direction}"
                cluster_result["criteria"][criterion_key] = serialized

            result[cluster_idx] = cluster_result

        return result
