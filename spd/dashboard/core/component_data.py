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


@dataclass
class GlobalMetrics:
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


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class TopKSample(SerializableDataclass):
    """A single top-k activating sample with embedded tokens."""

    tokens: list[int]  # Token IDs for this sample
    activations: Float[np.ndarray, " n_ctx"]  # Activation values across context
    max_activation: float  # Max activation in this sample
    mean_activation: float  # Mean activation in this sample


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
    coactivations: Float[np.ndarray, "n_alive n_alive"]
    correlations: Float[np.ndarray, "n_alive n_alive"]
    alive_component_labels: list[str]  # Index for matrices

    # Per-component stats (all components)
    components: list[SubcomponentStats]
