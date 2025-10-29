"""Minimal data structures for component-focused dashboard.

This module provides a clean two-phase architecture:
1. Generate raw activations (then delete model)
2. Process activations to compute global metrics and per-component stats
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass


@dataclass
class RawActivationData:
    """Pure activation data after model is deleted.

    Contains all raw activations and token data, with dead components filtered.
    """

    tokens: Float[np.ndarray, "n_samples n_ctx"]
    activations: dict[str, Float[np.ndarray, "n_samples n_ctx"]]
    component_labels: list[str]  # All components
    alive_components: list[str]  # Filtered by dead_threshold
    dead_components: set[str]  # Components with max activation <= threshold

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return self.tokens.shape[0]

    @property
    def n_ctx(self) -> int:
        """Context length."""
        return self.tokens.shape[1]


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class GlobalMetrics(SerializableDataclass):
    """Global metrics computed once across all alive components.

    All matrices are indexed by `component_labels` list.
    """

    coactivations: Float[np.ndarray, "n_alive n_alive"]
    correlations: Float[np.ndarray, "n_alive n_alive"]
    embeddings: Float[np.ndarray, "n_alive embed_dim"]
    component_labels: list[str]  # Ordering for indexing matrices

    def get_embedding(self, label: str) -> Float[np.ndarray, " embed_dim"]:
        """Get embedding vector for a specific component."""
        idx: int = self.component_labels.index(label)
        return self.embeddings[idx]

    @classmethod
    def generate(
        cls,
        activations: dict[str, Float[np.ndarray, "n_samples n_ctx"]],
        component_labels: list[str],
        embed_dim: int,
    ) -> "GlobalMetrics":
        """Generate global metrics from component activations.

        Args:
            activations: Dict mapping component labels to activation arrays
            component_labels: Ordered list of components to include
            embed_dim: Dimensionality of embeddings

        Returns:
            GlobalMetrics with coactivations, correlations, and embeddings
        """
        # Stack and flatten: [n_total_positions, n_components]
        stacked: Float[np.ndarray, "n_total n_components"] = np.stack(
            [activations[label].flatten() for label in component_labels], axis=1
        )

        # Compute binary coactivations
        binary: Float[np.ndarray, "n_total n_components"] = (stacked > 0).astype(np.float32)
        coactivations: Float[np.ndarray, "n_components n_components"] = binary.T @ binary

        # Compute Pearson correlations
        correlations: Float[np.ndarray, "n_components n_components"] = np.corrcoef(stacked.T)

        # Compute Isomap embeddings from affinity
        affinity: Float[np.ndarray, "n_comp n_comp"] = coactivations + np.abs(correlations)
        max_affinity: float = float(affinity.max())
        distance: Float[np.ndarray, "n_comp n_comp"] = max_affinity - affinity
        np.fill_diagonal(distance, 0.0)
        distance = (distance + distance.T) / 2.0

        from sklearn.manifold import Isomap

        isomap: Isomap = Isomap(n_components=embed_dim, metric="precomputed")
        embeddings: Float[np.ndarray, "n_comp embed_dim"] = isomap.fit_transform(distance)

        return cls(  # pyright: ignore[reportCallIssue]
            coactivations=coactivations,
            correlations=correlations,
            embeddings=embeddings,
            component_labels=component_labels,
        )


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class TopKSample(SerializableDataclass):
    """A single top-k activating sample with embedded tokens."""

    token_strs: list[str]  # Token strings for this sample
    activations: Float[np.ndarray, " n_ctx"]  # Activation values across context


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class SubcomponentStats(SerializableDataclass):
    """Complete statistics for a single component.

    Contains embedding, top samples, global stats, and histograms.
    """

    # Identity
    label: str
    is_dead: bool

    # Embedding (None if dead)
    embedding: Float[np.ndarray, " embed_dim"] | None

    # Top activating samples (by different criteria)
    top_max: list[TopKSample]  # Top-k by max activation
    top_mean: list[TopKSample]  # Top-k by mean activation

    # Global statistics
    stats: dict[str, float]  # mean, std, min, max, median, q05, q25, q75, q95

    # Multiple histograms for different views
    histograms: dict[str, dict[str, list[float]]]  # {histogram_name: {counts: [...], edges: [...]}}


@serializable_dataclass(kw_only=True)  # pyright: ignore[reportUntypedClassDecorator]
class ComponentDashboardData(SerializableDataclass):
    """Complete dashboard data for ZANJ serialization.

    Self-contained structure with all data needed for visualization.
    """

    # Metadata
    model_path: str
    n_samples: int
    n_ctx: int
    n_components: int
    n_alive: int
    n_dead: int

    # Global metrics (alive components only)
    global_metrics: GlobalMetrics

    # Per-component stats (all components)
    components: list[SubcomponentStats]
