"""Minimal data structures for component-focused dashboard.

This module provides a clean two-phase architecture:
1. Generate raw activations (then delete model)
2. Process activations to compute global metrics and per-component stats
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field

from spd.dashboard.core.activations import SubcomponentLabel


@dataclass
class RawActivationData:
    """Pure activation data after model is deleted.

    Contains all raw activations and token data, with dead components filtered.
    """

    tokens: Float[np.ndarray, "n_samples n_ctx"]
    activations: dict[SubcomponentLabel, Float[np.ndarray, "n_samples n_ctx"]]
    component_labels: list[SubcomponentLabel]  # All components
    alive_components: list[SubcomponentLabel]  # Filtered by dead_threshold
    dead_components: set[SubcomponentLabel]  # Components with max activation <= threshold

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
    component_labels: list[str]  # Ordering for indexing matrices (as strings for serialization)

    def get_embedding(self, label: SubcomponentLabel | str) -> Float[np.ndarray, " embed_dim"]:
        """Get embedding vector for a specific component."""
        label_str: str = label.to_string() if isinstance(label, SubcomponentLabel) else label
        idx: int = self.component_labels.index(label_str)
        return self.embeddings[idx]

    @classmethod
    def generate(
        cls,
        activations: dict[SubcomponentLabel, Float[np.ndarray, "n_samples n_ctx"]],
        component_labels: list[SubcomponentLabel],
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
        # Convert labels to strings for serialization
        component_labels_str: list[str] = [label.to_string() for label in component_labels]

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
            component_labels=component_labels_str,
        )


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class TopKSample(SerializableDataclass):
    """A single top-k activating sample with embedded tokens."""

    token_strs: list[str]  # Token strings for this sample
    activations: Float[np.ndarray, " n_ctx"]  # Activation values across context


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class SubcomponentStats(SerializableDataclass):
    """Complete statistics for a single alive component.

    Contains embedding, top samples, global stats, and histograms.
    """

    # Identity
    label: str

    # Embedding
    embedding: Float[np.ndarray, " embed_dim"]

    # Top activating samples (by different criteria)
    top_max: list[TopKSample] = serializable_field(
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TopKSample.load(s) for s in x],
    )
    top_mean: list[TopKSample] = serializable_field(
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TopKSample.load(s) for s in x],
    )

    # Global statistics
    stats: dict[str, float]  # mean, std, min, max, median, q05, q25, q75, q95

    # Multiple histograms for different views
    histograms: dict[str, dict[str, list[float]]]  # {histogram_name: {counts: [...], edges: [...]}}}

    @classmethod
    def generate(
        cls,
        label: SubcomponentLabel,
        activations: Float[np.ndarray, "n_samples n_ctx"],
        tokens: Float[np.ndarray, "n_samples n_ctx"],
        tokenizer: any,
        global_metrics: "GlobalMetrics",
        k: int,
        hist_bins: int,
    ) -> "SubcomponentStats":
        """Generate complete statistics for a single alive component.

        Args:
            label: Component label
            activations: Activation values
            tokens: Token IDs
            tokenizer: Tokenizer to decode token IDs
            global_metrics: Precomputed global metrics
            k: Number of top samples to track
            hist_bins: Number of histogram bins

        Returns:
            SubcomponentStats with all computed statistics
        """
        from typing import Any

        from spd.dashboard.core.tokenization import simple_batch_decode

        # Compute top-k samples (max criterion)
        scores_max: Float[np.ndarray, " n_samples"] = activations.max(axis=1)
        top_k_idx_max: Float[np.ndarray, " k"] = np.argpartition(scores_max, -k)[-k:]
        top_k_idx_max = top_k_idx_max[np.argsort(scores_max[top_k_idx_max])][::-1]

        # Decode all top-k tokens at once using fast batch decode
        top_k_tokens_max: Float[np.ndarray, "k n_ctx"] = tokens[top_k_idx_max.astype(int)]
        top_k_strs_max: np.ndarray = simple_batch_decode(tokenizer, top_k_tokens_max)

        top_max: list[TopKSample] = []
        for i, idx in enumerate(top_k_idx_max):
            idx_int: int = int(idx)
            top_max.append(
                TopKSample(  # pyright: ignore[reportCallIssue]
                    token_strs=top_k_strs_max[i].tolist(),
                    activations=activations[idx_int],
                )
            )

        # Compute top-k samples (mean criterion)
        scores_mean: Float[np.ndarray, " n_samples"] = activations.mean(axis=1)
        top_k_idx_mean: Float[np.ndarray, " k"] = np.argpartition(scores_mean, -k)[-k:]
        top_k_idx_mean = top_k_idx_mean[np.argsort(scores_mean[top_k_idx_mean])][::-1]

        # Decode all top-k tokens at once using fast batch decode
        top_k_tokens_mean: Float[np.ndarray, "k n_ctx"] = tokens[top_k_idx_mean.astype(int)]
        top_k_strs_mean: np.ndarray = simple_batch_decode(tokenizer, top_k_tokens_mean)

        top_mean: list[TopKSample] = []
        for i, idx in enumerate(top_k_idx_mean):
            idx_int: int = int(idx)
            top_mean.append(
                TopKSample(  # pyright: ignore[reportCallIssue]
                    token_strs=top_k_strs_mean[i].tolist(),
                    activations=activations[idx_int],
                )
            )

        # Compute global statistics
        flat: Float[np.ndarray, " n_total"] = activations.flatten()
        stats: dict[str, float] = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "median": float(np.median(flat)),
            "q05": float(np.quantile(flat, 0.05)),
            "q25": float(np.quantile(flat, 0.25)),
            "q75": float(np.quantile(flat, 0.75)),
            "q95": float(np.quantile(flat, 0.95)),
        }

        # Helper to create histogram dict
        def _make_histogram(data: Float[np.ndarray, " n"], name: str) -> None:
            counts, edges = np.histogram(data, bins=hist_bins)
            histograms[name] = {
                "counts": counts.tolist(),
                "edges": edges.tolist(),
            }

        # Compute histograms
        histograms: dict[str, dict[str, list[float]]] = {}
        _make_histogram(flat, "all_activations")
        _make_histogram(activations.max(axis=1), "max_per_sample")
        _make_histogram(activations.mean(axis=1), "mean_per_sample")

        return cls(  # pyright: ignore[reportCallIssue]
            label=label.to_string(),
            embedding=global_metrics.get_embedding(label),
            top_max=top_max,
            top_mean=top_mean,
            stats=stats,
            histograms=histograms,
        )


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
    components: list[SubcomponentStats] = serializable_field(
        serialization_fn=lambda x: [c.serialize() for c in x],
        deserialize_fn=lambda x: [SubcomponentStats.load(c) for c in x],
    )
