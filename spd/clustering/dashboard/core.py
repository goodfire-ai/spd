"""Data structures for tracking max-activating text samples."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, NewType

import numpy as np
from jaxtyping import Float

# Type aliases
TextSampleHash = NewType("TextSampleHash", str)
ActivationSampleHash = NewType("ActivationSampleHash", str)
ClusterIdHash = NewType("ClusterIdHash", str)
TrackingCriterionHash = NewType("TrackingCriterionHash", str)
ClusterLabel = NewType("ClusterLabel", int)  # Just a cluster index
Direction = Literal["max", "min"]

_SEPARATOR_1: str = "-"
_SEPARATOR_2: str = ":"


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentInfo:
    """Component information from merge history."""

    module: str
    index: int

    @property
    def label(self) -> str:
        """Component label as 'module:index'."""
        return f"{self.module}:{self.index}"


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
        assert all(_SEPARATOR_1 not in part for part in parts_str), "Parts cannot contain separator"
        assert all(_SEPARATOR_2 not in part for part in parts_str), "Parts cannot contain separator"

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
        """Hashes uniquely identifying each activation sample (cluster_hash + text_hash)."""
        cluster_str = self.cluster_id.to_string()
        return [ActivationSampleHash(f"{cluster_str}{_SEPARATOR_2}{th}") for th in self.text_hashes]

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
            direction=direction,  # type: ignore[arg-type]
            n_samples=int(parts[2]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BinnedData:
    bin_edges: list[float]
    bin_counts: list[int]

    @classmethod
    def from_arr(cls, arr: np.ndarray, n_bins: int) -> "BinnedData":
        """Create BinnedData from a numpy array."""
        counts, edges = np.histogram(arr, bins=n_bins)
        return cls(bin_edges=edges.tolist(), bin_counts=counts.tolist())


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterData:
    cluster_hash: ClusterIdHash
    components: list[ComponentInfo]  # Component info: module, index, label
    criterion_samples: dict[TrackingCriterionHash, list[TextSampleHash]]
    stats: dict[str, Any]

    @classmethod
    def generate(
        cls,
        cluster_id: ClusterId,
        activation_samples: ActivationSampleBatch,
        criteria: list[TrackingCriterion],
        components: list[ComponentInfo],
        hist_bins: int = 10,
        activation_threshold: float = 0.5,
        top_n_tokens: int = 50,
    ) -> "ClusterData":
        cluster_hash: ClusterIdHash = cluster_id.to_string()
        criterion_samples: dict[TrackingCriterionHash, list[TextSampleHash]] = {}
        stats: dict[str, Any] = dict()

        # filter out top-k samples per criterion
        for criterion in criteria:
            # Extract property values
            prop_values: Float[np.ndarray, " batch"] = ACTIVATION_SAMPLE_BATCH_STATS[
                criterion.property_name
            ](activation_samples)
            # Sort by property value
            reverse: bool = criterion.direction == "max"

            # Zip property values with text hashes and sort
            samples_with_values: list[tuple[TextSampleHash, float]] = list(
                zip(activation_samples.text_hashes, prop_values.tolist(), strict=True)
            )
            samples_with_values.sort(key=lambda x: x[1], reverse=reverse)

            # Take top k
            top_k: list[tuple[TextSampleHash, float]] = samples_with_values[: criterion.n_samples]

            # Extract just text hashes
            criterion_samples[criterion.to_string()] = [th for th, _ in top_k]

            # add stats
            stats[criterion.to_string()] = BinnedData.from_arr(
                prop_values,
                n_bins=hist_bins,
            )

        # Add general stats
        all_activations: Float[np.ndarray, " batch n_ctx"] = activation_samples.activations
        stats["all_activations"] = BinnedData.from_arr(
            all_activations.flatten(),
            n_bins=hist_bins,
        )
        stats["n_samples"] = len(activation_samples)
        stats["n_tokens"] = int(all_activations.size)
        stats["mean_activation"] = float(np.mean(all_activations))
        stats["min_activation"] = float(np.min(all_activations))
        stats["max_activation"] = float(np.max(all_activations))
        stats["median_activation"] = float(np.median(all_activations))

        # Token-level activation statistics
        if activation_samples.tokens is not None:
            from collections import Counter

            # Count activations per token above threshold
            token_activation_counts: Counter[str] = Counter()

            for sample_idx, token_list in enumerate(activation_samples.tokens):
                sample_activations = all_activations[sample_idx]  # shape: (n_ctx,)

                for token_idx, token in enumerate(token_list):
                    if sample_activations[token_idx] > activation_threshold:
                        token_activation_counts[token] += 1

            # Get top N most frequently activated tokens
            top_tokens: list[tuple[str, int]] = token_activation_counts.most_common(top_n_tokens)

            # Distribution statistics
            total_unique_tokens: int = len(token_activation_counts)
            total_activations: int = sum(token_activation_counts.values())

            # Compute concentration metrics
            if total_activations > 0 and total_unique_tokens > 0:
                # Entropy: measures how evenly distributed activations are across tokens
                counts_array: np.ndarray = np.array(list(token_activation_counts.values()))
                probs: np.ndarray = counts_array / total_activations
                entropy: float = float(-np.sum(probs * np.log(probs + 1e-10)))

                # Concentration ratio: fraction of activations in top 10% of tokens
                top_10pct_count: int = max(1, total_unique_tokens // 10)
                top_10pct_activations: int = sum(
                    count for _, count in token_activation_counts.most_common(top_10pct_count)
                )
                concentration_ratio: float = (
                    top_10pct_activations / total_activations if total_activations > 0 else 0.0
                )
            else:
                entropy = 0.0
                concentration_ratio = 0.0

            stats["token_activations"] = {
                "top_tokens": [{"token": token, "count": count} for token, count in top_tokens],
                "total_unique_tokens": total_unique_tokens,
                "total_activations": total_activations,
                "entropy": entropy,
                "concentration_ratio": concentration_ratio,
                "activation_threshold": activation_threshold,
            }

        return cls(
            cluster_hash=cluster_hash,
            components=components,
            criterion_samples=criterion_samples,
            stats=stats,
        )

    def get_unique_text_hashes(self) -> set[TextSampleHash]:
        """Get all unique text hashes across all criteria."""
        unique_hashes: set[TextSampleHash] = set()
        for hashes in self.criterion_samples.values():
            unique_hashes.update(hashes)
        return unique_hashes

    def get_unique_activation_hashes(self) -> set[ActivationSampleHash]:
        """Get all unique activation hashes across all criteria."""
        unique_hashes: set[ActivationSampleHash] = set()
        cluster_str = self.cluster_hash
        for hashes in self.criterion_samples.values():
            unique_hashes.update(
                ActivationSampleHash(f"{cluster_str}{_SEPARATOR_2}{th}") for th in hashes
            )
        return unique_hashes

    def serialize(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        # Convert stats to JSON-compatible format
        serialized_stats: dict[str, Any] = {}
        for key, value in self.stats.items():
            if isinstance(value, BinnedData):
                serialized_stats[key] = {
                    "bin_edges": value.bin_edges,
                    "bin_counts": value.bin_counts,
                }
            elif isinstance(value, int | float):
                serialized_stats[key] = value
            elif isinstance(value, np.ndarray):
                serialized_stats[key] = value.tolist()
            else:
                serialized_stats[key] = value

        return {
            "cluster_hash": self.cluster_hash,
            "components": [
                {"module": comp.module, "index": comp.index, "label": comp.label}
                for comp in self.components
            ],
            "criterion_samples": {
                str(k): [str(h) for h in v] for k, v in self.criterion_samples.items()
            },
            "stats": serialized_stats,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DashboardData:
    """All data for the dashboard."""

    clusters: dict[ClusterIdHash, ClusterData]
    text_samples: dict[TextSampleHash, TextSample]
    activations_map: dict[ActivationSampleHash, int]
    activations: ActivationSampleBatch

    # activations_map maps ActivationSampleHash to index in `activations`

    def save(self, output_dir: str) -> None:
        """Save dashboard data to directory structure for efficient frontend access.

        Structure:
        - clusters.json - All cluster data
        - text_samples.json - All text samples by hash
        - activations.npz - Numpy array with all activations
        - activations_map.json - Maps activation hashes to indices in activations array
        """
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save all cluster data in one file
        clusters_serialized = {
            str(cluster_hash): cluster_data.serialize()
            for cluster_hash, cluster_data in self.clusters.items()
        }
        with open(output_path / "clusters.json", "w") as f:
            json.dump(clusters_serialized, f, indent=2)

        # Save text samples
        text_samples_serialized = {
            str(hash_): {
                "full_text": sample.full_text,
                "tokens": sample.tokens,
                "text_hash": str(sample.text_hash),
            }
            for hash_, sample in self.text_samples.items()
        }
        with open(output_path / "text_samples.json", "w") as f:
            json.dump(text_samples_serialized, f, indent=2)

        # Save activations array as .npy for JavaScript
        np.save(output_path / "activations.npy", self.activations.activations)

        # Save activations map
        activations_map_serialized = {
            str(hash_): idx for hash_, idx in self.activations_map.items()
        }
        with open(output_path / "activations_map.json", "w") as f:
            json.dump(activations_map_serialized, f, indent=2)
