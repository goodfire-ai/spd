import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field


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

    label: str
    embedding: Float[np.ndarray, " embed_dim"]
    stats: dict[str, float]
    histograms: dict[str, dict[str, list[float]]]

    # Token statistics needed for table display (unified list with both probabilities)
    token_stats: list[TokenStat] = serializable_field(
        default_factory=list,
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [TokenStat.load(s) for s in x],
    )
