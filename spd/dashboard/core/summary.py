import warnings
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field

from spd.dashboard.core.acts import ComponentLabel
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig
from spd.dashboard.core.toks import TokenSequenceData


def _make_histogram(
    data: Float[np.ndarray, " n"],
    hist_bins: int = 10,
    hist_range: tuple[float, float] | None = None,
    round_bins: int | None = 1,
) -> dict[Literal["counts", "edges"], list[float]]:
    # Check for values outside range and warn if found
    hist_range_: tuple[float, float]
    if hist_range is not None:
        range_min, range_max = hist_range
        outside_range = (data < range_min) | (data > range_max)
        n_outside = int(np.sum(outside_range))
        if n_outside > 0:
            data_min, data_max = float(np.min(data)), float(np.max(data))
            warnings.warn(
                f"Data contains {n_outside} values outside histogram range "
                + f"[{range_min}, {range_max}] (data min: {data_min}, max: {data_max})",
                stacklevel=2,
            )
        hist_range_ = hist_range
    else:
        hist_range_ = (float(np.min(data)), float(np.max(data)))

    if round_bins is not None:
        hist_range_ = (
            round(hist_range_[0], round_bins) - (0.1**round_bins),
            round(hist_range_[1], round_bins) + (0.1**round_bins),
        )

    counts, edges = np.histogram(data, bins=hist_bins, range=hist_range)
    return {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
    }


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class TokenStat(SerializableDataclass):
    """Statistics for a single token's relationship with a component."""

    token: str
    token_id: int
    p_token_given_active: float  # P(token=X | component active)
    p_active_given_token: float  # P(component active | token=X)
    count_when_active: int  # Co-occurrence count
    count_token_total: int  # Total occurrences of token in dataset


def _prefix_dict(d: dict[str, Any], prefix: str, sep: str = ".") -> dict[str, Any]:
    """Prefix all keys in a dict with a given string."""
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class SubcomponentSummary(SerializableDataclass):
    """Lightweight summary for index.html table display.

    Contains only the data needed by the main component table,
    excluding large fields like top_samples.
    """

    label: ComponentLabel = serializable_field(
        serialization_fn=lambda x: x.serialize(),
        deserialize_fn=lambda x: ComponentLabel(module=x.split(":")[0], index=int(x.split(":")[1])),
    )
    embedding: Float[np.ndarray, " embed_dim"]
    stats: dict[str, float]
    histograms: dict[str, dict[str, list[float]]]

    # Token statistics needed for table display (unified list with both probabilities)
    token_stats: list[TokenStat] = serializable_field(
        default_factory=list,
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TokenStat.load(s) for s in x],
    )

    def to_embed_row(self) -> dict[str, Any]:
        """Convert to dict row for embedding visualization page"""
        return {
            **_prefix_dict(
                {
                    "module": self.label.module,
                    "component_index": self.label.index,
                    "label": self.label.as_str(),
                    "layer": int(
                        self.label.module.split(".")[2]
                    ),  # assuming module names like 'model.layers.{i}.*'
                },
                prefix="meta",
            ),
            **_prefix_dict(self.stats, prefix="stats"),
            **_prefix_dict(
                {str(i): float(self.embedding[i]) for i in range(self.embedding.shape[0])},
                prefix="embed.d3",
            ),
        }

    @classmethod
    def create(
        cls,
        label: ComponentLabel,
        tokens: TokenSequenceData,
        activations: Float[np.ndarray, " n_samples"],
        embeds: Float[np.ndarray, " embed_dim"],
        config: ComponentDashboardConfig,
        p_active_given_token: Float[np.ndarray, " d_vocab"],
        p_token_given_active: Float[np.ndarray, " d_vocab"],
        activated_per_token: Float[np.ndarray, " d_vocab"],
    ) -> "SubcomponentSummary":
        """Create summary for a single component."""

        # Compute basic stats
        stats: dict[str, float] = {
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "min": float(np.min(activations)),
            "max": float(np.max(activations)),
            "median": float(np.median(activations)),
        }

        # Compute histograms
        # Reshape to (n_batches, n_ctx) for per-sample statistics
        n_batches: int
        n_ctx: int
        n_batches, n_ctx = tokens.tokens.shape
        acts_2d: Float[np.ndarray, "n_batches n_ctx"] = activations.reshape(n_batches, n_ctx)

        histograms: dict[str, dict[str, list[float]]] = {
            "all_activations": _make_histogram(activations, config.hist_bins, config.hist_range),
            "max_per_sample": _make_histogram(
                acts_2d.max(axis=1), config.hist_bins, config.hist_range
            ),
            "mean_per_sample": _make_histogram(
                acts_2d.mean(axis=1), config.hist_bins, config.hist_range
            ),
        }

        # Compute token stats
        top_n: int = config.token_stats_top_n

        # Get top K indices by each probability metric
        top_by_p_token_given_active: Int[np.ndarray, " top_n"] = np.argsort(p_token_given_active)[
            -top_n:
        ]
        top_by_p_active_given_token: Int[np.ndarray, " top_n"] = np.argsort(p_active_given_token)[
            -top_n:
        ]

        # Union of both sets
        token_indices: Int[np.ndarray, " n_indices"] = np.unique(
            np.concatenate([top_by_p_token_given_active, top_by_p_active_given_token])
        )

        # Create TokenStat objects
        token_stats_list: list[TokenStat] = []
        idx: int
        for idx in token_indices:
            token_str: str = tokens.vocab_arr[idx]
            # Count total occurrences of this token in the dataset
            count_token_total: int = int(np.sum(tokens.tokens.reshape(-1) == idx))

            token_stat: TokenStat = TokenStat(
                token=token_str,
                token_id=int(idx),
                p_token_given_active=float(p_token_given_active[idx]),
                p_active_given_token=float(p_active_given_token[idx]),
                count_when_active=int(activated_per_token[idx]),
                count_token_total=count_token_total,
            )
            token_stats_list.append(token_stat)

        # Sort by max of the two probabilities
        token_stats_list.sort(
            key=lambda ts: max(ts.p_token_given_active, ts.p_active_given_token),
            reverse=True,
        )

        return cls(
            label=label,
            embedding=embeds,
            stats=stats,
            histograms=histograms,
            token_stats=token_stats_list,
        )
