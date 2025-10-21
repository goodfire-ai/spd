"""Cluster-specific data structures for dashboard."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float

from spd.clustering.consts import SubComponentKey
from spd.clustering.dashboard.core.base import (
    _SEPARATOR_1,
    _SEPARATOR_3,
    ACTIVATION_SAMPLE_BATCH_STATS,
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterId,
    ClusterIdHash,
    ClusterSample,
    Direction,
    TextSampleHash,
    TrackingCriterionHash,
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
    components: list[SubComponentKey]  # Component info: module, index, label
    samples: list[ClusterSample]  # Self-contained samples with text + activations
    stats: dict[str, Any]
    # Component-level metrics
    component_coactivations: Float[np.ndarray, "n_comps n_comps"] | None = None
    # TODO: Add component_cosine_similarities for U/V vectors when dimension mismatch is resolved

    # DEPRECATED: Can be removed once refactor is complete and JS is updated
    criterion_samples: dict[TrackingCriterionHash, list[TextSampleHash]] | None = None

    @classmethod
    def generate(
        cls,
        cluster_id: ClusterId,
        activation_samples: ActivationSampleBatch,
        criteria: list[TrackingCriterion],
        components: list[SubComponentKey],
        hist_bins: int = 10,
        activation_threshold: float = 0.5,
        top_n_tokens: int = 50,
    ) -> "ClusterData":
        cluster_hash: ClusterIdHash = cluster_id.to_string()
        stats: dict[str, Any] = dict()

        # Build mapping: text_hash -> (sample_idx, criteria_list)
        sample_criteria_map: dict[TextSampleHash, list[str]] = {}
        sample_indices: dict[TextSampleHash, int] = {}

        # DEPRECATED: For backward compatibility during transition
        criterion_samples_deprecated: dict[TrackingCriterionHash, list[TextSampleHash]] = {}

        # Filter out top-k samples per criterion
        for criterion in criteria:
            # Extract property values
            prop_values: Float[np.ndarray, " batch"] = ACTIVATION_SAMPLE_BATCH_STATS[
                criterion.property_name
            ](activation_samples)
            # Sort by property value
            reverse: bool = criterion.direction == "max"

            # Zip property values with text hashes and indices
            samples_with_values: list[tuple[int, TextSampleHash, float]] = [
                (idx, th, val)
                for idx, (th, val) in enumerate(zip(activation_samples.text_hashes, prop_values.tolist(), strict=True))
            ]
            samples_with_values.sort(key=lambda x: x[2], reverse=reverse)

            # Take top k
            top_k: list[tuple[int, TextSampleHash, float]] = samples_with_values[: criterion.n_samples]

            criterion_str = criterion.to_string()

            # Track which samples satisfy which criteria
            for idx, text_hash, _ in top_k:
                sample_criteria_map.setdefault(text_hash, []).append(criterion_str)
                sample_indices[text_hash] = idx

            # DEPRECATED: Keep for backward compat
            criterion_samples_deprecated[criterion.to_string()] = [th for _, th, _ in top_k]

            # Add stats
            stats[criterion_str] = BinnedData.from_arr(
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

        # Compute max activation position distribution (how concentrated are the max activations)
        # For each sample, find the position (index) where max activation occurs
        max_positions: np.ndarray = np.argmax(all_activations, axis=1)  # shape: (batch,)
        # Normalize positions to [0, 1] range
        n_ctx: int = all_activations.shape[1]
        normalized_positions: np.ndarray = max_positions.astype(float) / max(1, n_ctx - 1)

        # Sanity check: positions should always be in [0, 1]
        assert normalized_positions.min() >= 0, f"Position min={normalized_positions.min()} < 0"
        assert normalized_positions.max() <= 1, f"Position max={normalized_positions.max()} > 1"

        stats["max_activation_position"] = BinnedData.from_arr(
            normalized_positions,
            n_bins=hist_bins,
        )

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

        # Build self-contained ClusterSample objects
        samples: list[ClusterSample] = []
        for text_hash, criteria_list in sample_criteria_map.items():
            sample_idx = sample_indices[text_hash]
            samples.append(
                ClusterSample(
                    text_hash=str(text_hash),
                    tokens=activation_samples.tokens[sample_idx] if activation_samples.tokens else [],
                    activations=activation_samples.activations[sample_idx],
                    criteria=criteria_list,
                )
            )

        return cls(
            cluster_hash=cluster_hash,
            components=components,
            samples=samples,
            stats=stats,
            criterion_samples=criterion_samples_deprecated,  # DEPRECATED
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
                ActivationSampleHash(f"{cluster_str}{_SEPARATOR_3}{th}") for th in hashes
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

        result: dict[str, Any] = {
            "cluster_hash": self.cluster_hash,
            "components": [
                {"module": comp.module, "index": comp.index, "label": comp.label}
                for comp in self.components
            ],
            # Serialize samples (ClusterSample has .serialize() from SerializableDataclass)
            "samples": [sample.serialize() for sample in self.samples],
            "stats": serialized_stats,
        }

        # DEPRECATED: Keep for backward compat during transition
        if self.criterion_samples is not None:
            result["criterion_samples"] = {
                str(k): [str(h) for h in v] for k, v in self.criterion_samples.items()
            }

        # Add component-level metrics if available
        if self.component_coactivations is not None:
            result["component_coactivations"] = self.component_coactivations.tolist()
        # TODO: Serialize component_cosine_similarities when implemented

        return result
