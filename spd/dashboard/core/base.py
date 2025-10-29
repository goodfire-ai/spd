"""Foundational data structures and type aliases for dashboard."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NewType

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass

# Type aliases
TextSampleHash = NewType("TextSampleHash", str)
ActivationSampleHash = NewType("ActivationSampleHash", str)
ClusterIdHash = NewType("ClusterIdHash", str)
TrackingCriterionHash = NewType("TrackingCriterionHash", str)
ClusterLabel = NewType("ClusterLabel", int)  # Just a cluster index
Direction = Literal["max", "min"]

_SEPARATOR_1: str = "."  # For cluster_id parts (e.g., runid-iteration-label)
_SEPARATOR_2: str = ":"  # ONLY for component labels (e.g., module:component_index)
_SEPARATOR_3: str = "|"  # For ALL activation hashes (cluster|text or cluster|comp|text)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterId:
    """Unique identifier for a cluster. This should uniquely identify a cluster *globally*"""

    clustering_run: str  # Clustering run identifier
    iteration: int  # Merge iteration number
    cluster_label: ClusterLabel  # Cluster index

    def to_tuple(self) -> tuple[str, int, int]:
        """Return as a tuple of identifying components."""
        return (
            self.clustering_run,
            self.iteration,
            self.cluster_label,
        )

    def to_string(self) -> ClusterIdHash:
        """Hash uniquely identifying this cluster."""
        # Use all identifying information to create unique hash
        parts_tuple = self.to_tuple()
        parts_str = tuple(str(part) for part in parts_tuple)
        assert all(_SEPARATOR_1 not in part for part in parts_str), (
            f"Parts cannot contain separator {_SEPARATOR_1=}, {parts_tuple=}, {parts_str=}"
        )
        assert all(_SEPARATOR_2 not in part for part in parts_str), (
            f"Parts cannot contain separator {_SEPARATOR_2=}, {parts_tuple=}, {parts_str=}"
        )

        return ClusterIdHash(_SEPARATOR_1.join(parts_str))

    @classmethod
    def from_string(cls, s: ClusterIdHash) -> "ClusterId":
        """Create ClusterId from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_1)
        if len(parts) != 3:
            raise ValueError(f"Invalid ClusterId string: {s}")
        return cls(
            clustering_run=parts[0],
            iteration=int(parts[1]),
            cluster_label=ClusterLabel(int(parts[2])),
        )


@dataclass(frozen=True, kw_only=True)
class TextSample:
    """Text content, with a reference to the dataset. depends on tokenizer used."""

    full_text: str
    tokens: list[str]

    @cached_property
    def text_hash(self) -> TextSampleHash:
        """Hash of full_text for deduplication."""
        return TextSampleHash(hashlib.sha256(self.full_text.encode()).hexdigest()[:8])

    def length(self) -> int:
        """Return the number of tokens."""
        return len(self.tokens)


@dataclass(frozen=True, kw_only=True)
class ActivationSampleBatch:
    cluster_id: ClusterId
    text_hashes: list[TextSampleHash]
    activations: Float[np.ndarray, "batch n_ctx"]
    tokens: list[list[str]] | None = None  # Token strings for each sample

    @cached_property
    def activation_hashes(self) -> list[ActivationSampleHash]:
        """Hashes uniquely identifying each activation sample (cluster_hash | text_hash)."""
        cluster_str = self.cluster_id.to_string()
        return [ActivationSampleHash(f"{cluster_str}{_SEPARATOR_3}{th}") for th in self.text_hashes]

    @cached_property
    def activation_hashes_short(self) -> list[str]:
        """Short hashes for frontend (clusterLabel|textHash) without run ID and iteration."""
        cluster_label = str(self.cluster_id.cluster_label)
        return [f"{cluster_label}{_SEPARATOR_3}{th}" for th in self.text_hashes]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the activations array (batch_size, n_ctx)."""
        return self.activations.shape

    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        n_samples: int = self.activations.shape[0]
        assert len(self.text_hashes) == n_samples, "Mismatch between text_hashes and activations"
        return n_samples


ACTIVATION_SAMPLE_BATCH_STATS: dict[
    str, Callable[[ActivationSampleBatch], Float[np.ndarray, " batch"]]
] = dict(
    mean_activation=lambda batch: np.mean(batch.activations, axis=1),
    min_activation=lambda batch: np.min(batch.activations, axis=1),
    median_activation=lambda batch: np.median(batch.activations, axis=1),
    max_activation=lambda batch: np.max(batch.activations, axis=1),
    max_position=lambda batch: np.argmax(batch.activations, axis=1).astype(float),
)


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class ClusterSample(SerializableDataclass):
    """Self-contained sample combining text reference, tokens, and activations.

    This allows clusters to be self-contained without requiring external lookups.
    """

    text_hash: str  # Reference to TextSample in text_samples dict
    tokens: list[str]  # Token strings for display
    activations: Float[np.ndarray, " n_ctx"]  # ZANJ will save as .npy ref
    criteria: list[
        str
    ]  # Which tracking criteria this sample satisfied (e.g., ["max_activation-max-16"])
