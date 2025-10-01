"""Data structures for tracking max-activating text samples."""

import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NewType

import numpy as np

# Type aliases
TextSampleHash = NewType("TextSampleHash", str)
ActivationSampleHash = NewType("ActivationSampleHash", str)
ClusterIdHash = NewType("ClusterIdHash", str)
TrackingCriterionHash = NewType("TrackingCriterionHash", str)
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
        """Return as a string 'module_name-original_index'."""
        return f"{self.module_name}{_SEPARATOR_1}{self.original_index}"

    @staticmethod
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

    def to_tuple(self) -> tuple[str, str, int, str, int]:
        """Return as a tuple of identifying components."""
        return (
            self.spd_run,
            self.clustering_run,
            self.iteration,
            self.cluster_label.module_name,
            self.cluster_label.original_index,
        )

    def to_string(self) -> ClusterIdHash:
        """Hash uniquely identifying this cluster."""
        # Use all identifying information to create unique hash
        parts_tuple = self.to_tuple()
        parts_str = tuple(str(part) for part in parts_tuple)
        assert all(_SEPARATOR_1 not in part for part in parts_str), (
            "Parts cannot contain separator"
        )
        assert all(_SEPARATOR_2 not in part for part in parts_str), (
            "Parts cannot contain separator"
        )

        return ClusterIdHash(_SEPARATOR_1.join(parts_str))

    @classmethod
    def from_string(cls, s: ClusterIdHash) -> "ClusterId":
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
    def text_hash(self) -> TextSampleHash:
        """Hash of full_text for deduplication."""
        return TextSampleHash(hashlib.sha256(self.full_text.encode()).hexdigest()[:8])

    def length(self) -> int:
        """Return the number of tokens."""
        return len(self.tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationSample:
    """Activations for a text sample in a specific cluster."""

    cluster_id: ClusterId
    text_hash: TextSampleHash  # Reference to TextSample
    activations: np.ndarray  # Shape: (seq_len,)

    @cached_property
    def activation_hash(self) -> ActivationSampleHash:
        """Hash uniquely identifying this activation sample (cluster_hash + text_hash)."""
        # Combine cluster_hash with text_hash to create unique identifier
        return ActivationSampleHash(f"{self.cluster_id.to_string()}:{self.text_hash}")

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

    def to_tuple(self) -> tuple[str, Direction, int]:
        """Return as a tuple of (property_name, direction, n_samples)."""
        return (self.property_name, self.direction, self.n_samples)

    def to_string(self) -> TrackingCriterionHash:
        """Convert to hash string representation."""
        parts = (self.property_name, self.direction, str(self.n_samples))
        assert all(_SEPARATOR_1 not in part for part in parts), "Parts cannot contain separator"
        return TrackingCriterionHash(_SEPARATOR_1.join(parts))

    @classmethod
    def from_string(cls, s: TrackingCriterionHash) -> "TrackingCriterion":
        """Create TrackingCriterion from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_1)
        if len(parts) != 3:
            raise ValueError(f"Invalid TrackingCriterion string: {s}")
        direction = parts[1]
        if direction not in ("max", "min"):
            raise ValueError(f"Invalid direction: {direction}")
        return cls(
            property_name=parts[0],
            direction=direction,  # pyright: ignore[reportArgumentType]
            n_samples=int(parts[2]),
        )


# input data looks like `(cluster, text) -> activation` which is unique
# output data looks like:
# cluster -> {
#   {criterion -> texts}
#   {various stats about the cluster}
# }










































@dataclass
class ClusterCriterionTracker:
    """Accumulates all stat values per cluster per criterion, then filters to top-k"""

    all_samples: dict[ClusterIdHash, dict[TrackingCriterionHash, list[tuple[TextSampleHash, float]]]]

    @classmethod
    def create_empty(cls) -> "ClusterCriterionTracker":
        """Create an empty tracker."""
        return cls(all_samples={})

    def add_sample(
        self,
        cluster_hash: ClusterIdHash,
        text_hash: TextSampleHash,
        stat_values: dict[TrackingCriterionHash, float],
    ) -> None:
        """Add a sample to the tracker.

        Args:
            cluster_hash: Hash of the cluster
            text_hash: Hash of the text sample
            stat_values: Dict mapping TrackingCriterionHash to computed stat values
        """
        # Initialize cluster if needed
        if cluster_hash not in self.all_samples:
            self.all_samples[cluster_hash] = {}

        # Add to each criterion
        for crit_hash, stat_value in stat_values.items():
            # Initialize criterion if needed
            if crit_hash not in self.all_samples[cluster_hash]:
                self.all_samples[cluster_hash][crit_hash] = []

            # Add sample (allow duplicates for now, dedup can happen in filter_top_k)
            self.all_samples[cluster_hash][crit_hash].append((text_hash, stat_value))

    def filter_top_k(
        self, criteria: list[TrackingCriterion]
    ) -> dict[ClusterIdHash, dict[TrackingCriterionHash, list[TextSampleHash]]]:
        """Filter to top-k samples per cluster per criterion.

        Args:
            criteria: List of criteria specifying n_samples and direction

        Returns:
            Dict mapping cluster -> criterion -> list of top-k text hashes
        """
        criteria_map: dict[TrackingCriterionHash, TrackingCriterion] = {
            c.to_string(): c for c in criteria
        }

        result: dict[ClusterIdHash, dict[TrackingCriterionHash, list[TextSampleHash]]] = {}

        for cluster_hash, cluster_data in self.all_samples.items():
            result[cluster_hash] = {}

            for crit_hash, samples in cluster_data.items():
                if crit_hash not in criteria_map:
                    continue

                criterion = criteria_map[crit_hash]

                # Deduplicate by text_hash (keep first occurrence)
                seen: set[TextSampleHash] = set()
                deduped: list[tuple[TextSampleHash, float]] = []
                for text_hash, stat_value in samples:
                    if text_hash not in seen:
                        seen.add(text_hash)
                        deduped.append((text_hash, stat_value))

                # Sort by stat value
                reverse = criterion.direction == "max"
                sorted_samples = sorted(deduped, key=lambda x: x[1], reverse=reverse)

                # Take top k
                top_k = sorted_samples[: criterion.n_samples]

                # Extract just text hashes
                result[cluster_hash][crit_hash] = [th for th, _ in top_k]

        return result
