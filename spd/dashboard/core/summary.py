import warnings
from typing import Literal

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field

from spd.dashboard.core.acts import Activations, ComponentLabel, FlatActivations
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


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class SubcomponentSummary(SerializableDataclass):
    """Lightweight summary for index.html table display.

    Contains only the data needed by the main component table,
    excluding large fields like top_samples.
    """

    label: ComponentLabel
    embedding: Float[np.ndarray, " embed_dim"]
    stats: dict[str, float]
    histograms: dict[str, dict[str, list[float]]]

    # Token statistics needed for table display (unified list with both probabilities)
    token_stats: list[TokenStat] = serializable_field(
        default_factory=list,
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TokenStat.load(s) for s in x],
    )

    @classmethod
    def create(
        cls,
        label: ComponentLabel,
        tokens: TokenSequenceData,
        activations: Float[np.ndarray, " n_samples"],
        embeds: Float[np.ndarray, " embed_dim"],
    ) -> "SubcomponentSummary":
        """Create summary for a single component."""

        return cls(
            label=label,
            embedding=embeds,
            stats={},
            histograms={},
            token_stats=[],
        )


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class ActivationsSummary(SerializableDataclass):
    """Lightweight summary for index.html display.

    Contains only the data needed by the main page,
    excluding large fields like full activations.
    """

    summaries: list[SubcomponentSummary]

    @classmethod
    def from_activations(cls, activations: Activations) -> "ActivationsSummary":
        acts_flat: FlatActivations = FlatActivations.create(activations)
        assert acts_flat.n_components
        return ActivationsSummary([])
