"""Minimal data structures for component-focused dashboard.

This module provides a clean two-phase architecture:
1. Generate raw activations (then delete model)
2. Process activations to compute global metrics and per-component stats
"""

import warnings
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field

from spd.dashboard.core.activations import SubcomponentLabel
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig


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
class TokenStat(SerializableDataclass):
    """Statistics for a single token's relationship with a component."""

    token: str
    token_id: int
    p_token_given_active: float  # P(token=X | component active)
    p_active_given_token: float  # P(component active | token=X)
    count_when_active: int  # Co-occurrence count
    count_token_total: int  # Total occurrences of token in dataset


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class TokenActivationsSummary(SerializableDataclass):
    """Summary statistics for token activations.

    This structure matches what the dashboard JavaScript expects in stats.token_activations.
    """

    top_tokens: list[dict[str, str | int]]  # [{token: str, count: int}, ...]
    total_unique_tokens: int
    total_activations: int
    entropy: float
    concentration_ratio: float
    activation_threshold: float


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class SubcomponentSummary(SerializableDataclass):
    """Lightweight summary for index.html table display.

    Contains only the data needed by the main component table,
    excluding large fields like top_samples.
    """

    label: str
    embedding: Float[np.ndarray, " embed_dim"]
    stats: dict[str, float | dict]  # Basic stats + token_activations
    histograms: dict[str, dict[str, list[float]]]

    # Token statistics needed for table display (unified list with both probabilities)
    token_stats: list[TokenStat] = serializable_field(
        default_factory=list,
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TokenStat.load(s) for s in x],
    )


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class SubcomponentDetails(SerializableDataclass):
    """Complete statistics for a single alive component.

    Contains embedding, top samples, global stats, and histograms.
    """

    # Identity
    label: str

    # Embedding
    embedding: Float[np.ndarray, " embed_dim"]

    # Top activating samples (unified deduplicated list)
    # Union of samples that rank high by max OR mean activation
    top_samples: list[TopKSample] = serializable_field(
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TopKSample.load(s) for s in x],
    )

    # Global statistics
    stats: dict[str, float]  # mean, std, min, max, median, q05, q25, q75, q95

    # Multiple histograms for different views
    histograms: dict[
        str, dict[str, list[float]]
    ]  # {histogram_name: {counts: [...], edges: [...]}}}

    # Token activation summary (for dashboard JS)
    token_activations: TokenActivationsSummary

    # Detailed token statistics (unified list with both probabilities)
    token_stats: list[TokenStat] = serializable_field(
        default_factory=list,
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TokenStat.load(s) for s in x],
    )

    @classmethod
    def generate(
        cls,
        label: SubcomponentLabel,
        activations: Float[np.ndarray, "n_samples n_ctx"],
        tokens: Float[np.ndarray, "n_samples n_ctx"],
        tokenizer: any,
        global_metrics: "GlobalMetrics",
        config: "ComponentDashboardConfig",
    ) -> "SubcomponentDetails":
        """Generate complete statistics for a single alive component.

        Args:
            label: Component label
            activations: Activation values
            tokens: Token IDs
            tokenizer: Tokenizer to decode token IDs
            global_metrics: Precomputed global metrics
            config: Dashboard configuration with all parameters

        Returns:
            SubcomponentStats with all computed statistics
        """

        from spd.dashboard.core.tokenization import simple_batch_decode

        k: int = config.n_samples

        # Find top K samples by max activation criterion
        scores_max: Float[np.ndarray, " n_samples"] = activations.max(axis=1)
        top_k_idx_max: Float[np.ndarray, " k"] = np.argpartition(scores_max, -k)[-k:]
        top_k_idx_max = top_k_idx_max[np.argsort(scores_max[top_k_idx_max])][::-1]

        # Find top K samples by mean activation criterion
        scores_mean: Float[np.ndarray, " n_samples"] = activations.mean(axis=1)
        top_k_idx_mean: Float[np.ndarray, " k"] = np.argpartition(scores_mean, -k)[-k:]
        top_k_idx_mean = top_k_idx_mean[np.argsort(scores_mean[top_k_idx_mean])][::-1]

        # Union and deduplicate: combine both lists, keeping unique samples only
        # This gives us all samples that rank high by EITHER criterion
        all_indices_set: set[int] = set(top_k_idx_max.astype(int).tolist()) | set(
            top_k_idx_mean.astype(int).tolist()
        )
        all_unique_indices: list[int] = sorted(list(all_indices_set))

        # Batch decode all unique samples
        all_unique_tokens: Float[np.ndarray, "n_unique n_ctx"] = tokens[all_unique_indices]
        all_unique_strs: np.ndarray = simple_batch_decode(tokenizer, all_unique_tokens)

        # Build unified list of all top samples (stored in SubcomponentDetails)
        # Frontend or summary generation can later sort/filter by max or mean as needed
        all_top_samples: list[TopKSample] = []
        for i, idx in enumerate(all_unique_indices):
            all_top_samples.append(
                TopKSample(  # pyright: ignore[reportCallIssue]
                    token_strs=all_unique_strs[i].tolist(),
                    activations=activations[idx],
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
            # Check for values outside range and warn if found
            if config.hist_range is not None:
                range_min, range_max = config.hist_range
                outside_range = (data < range_min) | (data > range_max)
                n_outside = int(np.sum(outside_range))
                if n_outside > 0:
                    data_min, data_max = float(np.min(data)), float(np.max(data))
                    warnings.warn(
                        f"Histogram '{name}': {n_outside}/{len(data)} values outside range "
                        f"{config.hist_range} (data range: [{data_min:.6f}, {data_max:.6f}])",
                        UserWarning,
                        stacklevel=2,
                    )

            counts, edges = np.histogram(data, bins=config.hist_bins, range=config.hist_range)
            histograms[name] = {
                "counts": counts.tolist(),
                "edges": edges.tolist(),
            }

        # Compute histograms
        histograms: dict[str, dict[str, list[float]]] = {}
        _make_histogram(flat, "all_activations")
        _make_histogram(activations.max(axis=1), "max_per_sample")
        _make_histogram(activations.mean(axis=1), "mean_per_sample")

        # Compute token statistics
        # Flatten activations and tokens to [n_total_positions]
        flat_acts: Float[np.ndarray, " n_total"] = activations.flatten()
        flat_tokens: Float[np.ndarray, " n_total"] = tokens.flatten().astype(int)

        # Create binary mask for active positions
        is_active: Float[np.ndarray, " n_total"] = flat_acts > config.token_active_threshold

        # Count occurrences per token
        unique_tokens: Float[np.ndarray, " n_unique"] = np.unique(flat_tokens)
        token_counts: dict[int, int] = {}
        token_active_counts: dict[int, int] = {}

        for token_id in unique_tokens:
            token_id_int: int = int(token_id)
            token_mask: Float[np.ndarray, " n_total"] = flat_tokens == token_id_int
            token_counts[token_id_int] = int(np.sum(token_mask))
            token_active_counts[token_id_int] = int(np.sum(token_mask & is_active))

        # Compute P(token | active) for each token
        total_active: int = int(np.sum(is_active))
        p_token_given_active: dict[int, float] = {}
        if total_active > 0:
            for token_id, active_count in token_active_counts.items():
                p_token_given_active[token_id] = active_count / total_active

        # Compute P(active | token) for each token
        p_active_given_token: dict[int, float] = {}
        for token_id, total_count in token_counts.items():
            if total_count > 0:
                p_active_given_token[token_id] = token_active_counts[token_id] / total_count

        # Create TokenStat objects for all tokens
        all_token_stats: list[TokenStat] = []

        for token_id in unique_tokens:
            token_id_int: int = int(token_id)
            token_str: str = tokenizer.decode([token_id_int])

            token_stat: TokenStat = TokenStat(  # pyright: ignore[reportCallIssue]
                token=token_str,
                token_id=token_id_int,
                p_token_given_active=p_token_given_active.get(token_id_int, 0.0),
                p_active_given_token=p_active_given_token.get(token_id_int, 0.0),
                count_when_active=token_active_counts[token_id_int],
                count_token_total=token_counts[token_id_int],
            )

            all_token_stats.append(token_stat)

        # Get top N by each metric and create union
        top_by_token_given_active = sorted(
            all_token_stats, key=lambda x: x.p_token_given_active, reverse=True
        )[: config.token_stats_details_top_n]

        top_by_active_given_token = sorted(
            all_token_stats, key=lambda x: x.p_active_given_token, reverse=True
        )[: config.token_stats_details_top_n]

        # Create union using token_id as key
        token_stats_dict: dict[int, TokenStat] = {}
        for stat in top_by_token_given_active + top_by_active_given_token:
            token_stats_dict[stat.token_id] = stat

        # Convert to list and sort by max of the two probabilities
        unified_token_stats: list[TokenStat] = sorted(
            token_stats_dict.values(),
            key=lambda x: max(x.p_token_given_active, x.p_active_given_token),
            reverse=True,
        )

        # Create TokenActivationsSummary
        # Compute entropy of token distribution when active
        entropy: float = 0.0
        if total_active > 0:
            for prob in p_token_given_active.values():
                if prob > 0:
                    entropy -= prob * np.log2(prob)

        # Compute concentration ratio (proportion of activations from top 10 tokens)
        top_10_active_count: int = sum(sorted(token_active_counts.values(), reverse=True)[:10])
        concentration_ratio: float = top_10_active_count / total_active if total_active > 0 else 0.0

        # Format top tokens for summary (by P(token|active))
        top_by_p_token_active = sorted(
            unified_token_stats, key=lambda x: x.p_token_given_active, reverse=True
        )[: config.token_activations_summary_top_n]
        top_tokens_summary: list[dict[str, str | int]] = [
            {"token": stat.token, "count": stat.count_when_active} for stat in top_by_p_token_active
        ]

        token_activations_summary = TokenActivationsSummary(  # pyright: ignore[reportCallIssue]
            top_tokens=top_tokens_summary,
            total_unique_tokens=len(unique_tokens),
            total_activations=total_active,
            entropy=float(entropy),
            concentration_ratio=float(concentration_ratio),
            activation_threshold=config.token_active_threshold,
        )

        return cls(  # pyright: ignore[reportCallIssue]
            label=label.to_string(),
            embedding=global_metrics.get_embedding(label),
            top_samples=all_top_samples,  # Unified deduped list; frontend sorts by max/mean as needed
            stats=stats,
            histograms=histograms,
            token_activations=token_activations_summary,
            token_stats=unified_token_stats,
        )

    def to_summary(self, token_stats_top_n: int = 5) -> SubcomponentSummary:
        """Convert to lightweight summary for index.html table display.

        Args:
            token_stats_top_n: Number of top tokens to include by each metric

        Returns:
            SubcomponentSummary with basic stats + token_activations, excluding heavy fields like top_samples
        """
        # Merge basic stats with token_activations summary
        stats_with_tokens: dict[str, float | dict] = dict(self.stats)
        stats_with_tokens["token_activations"] = self.token_activations.serialize()

        # Filter token_stats to union of top N by each metric
        top_by_token_given_active = sorted(
            self.token_stats, key=lambda x: x.p_token_given_active, reverse=True
        )[:token_stats_top_n]

        top_by_active_given_token = sorted(
            self.token_stats, key=lambda x: x.p_active_given_token, reverse=True
        )[:token_stats_top_n]

        # Create union
        token_stats_dict: dict[int, TokenStat] = {}
        for stat in top_by_token_given_active + top_by_active_given_token:
            token_stats_dict[stat.token_id] = stat

        # Sort by max probability
        summary_token_stats: list[TokenStat] = sorted(
            token_stats_dict.values(),
            key=lambda x: max(x.p_token_given_active, x.p_active_given_token),
            reverse=True,
        )

        return SubcomponentSummary(  # pyright: ignore[reportCallIssue]
            label=self.label,
            embedding=self.embedding,
            stats=stats_with_tokens,
            histograms=self.histograms,
            token_stats=summary_token_stats,
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
    config: "ComponentDashboardConfig"  # For accessing configuration values

    # Global metrics (alive components only)
    global_metrics: GlobalMetrics

    # Per-component stats (all components)
    components: list[SubcomponentDetails] = serializable_field(
        serialization_fn=lambda x: [c.serialize() for c in x],
        deserialize_fn=lambda x: [SubcomponentDetails.load(c) for c in x],
    )

    @property
    def subcomponent_summaries(self) -> list[SubcomponentSummary]:
        """Lightweight summaries for index.html table display.

        Returns:
            List of SubcomponentSummary objects (excludes heavy fields like top_max, top_mean)
        """
        return [
            component.to_summary(token_stats_top_n=self.config.token_stats_top_n)
            for component in self.components
        ]

    @property
    def subcomponent_details(self) -> dict[str, list[SubcomponentDetails]]:
        """Component details grouped by module name for lazy loading.

        Returns:
            Dict mapping module names to their component lists
        """
        from collections import defaultdict

        grouped: dict[str, list[SubcomponentDetails]] = defaultdict(list)

        for component in self.components:
            # Extract module name from label (format: "module.name:index")
            module_name: str = component.label.rsplit(":", 1)[0]
            grouped[module_name].append(component)

        return dict(grouped)
