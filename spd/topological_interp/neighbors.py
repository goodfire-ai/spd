"""Gather neighbor context from attributions and correlations."""

from dataclasses import dataclass

from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.harvest.analysis import get_correlated_components
from spd.harvest.storage import CorrelationStorage
from spd.topological_interp.db import TopologicalInterpDB
from spd.topological_interp.ordering import is_later_layer, parse_component_key


@dataclass
class NeighborContext:
    component_key: str
    attribution: float
    label: str | None
    confidence: str | None
    jaccard: float | None
    pmi: float | None


def get_downstream_neighbors(
    component_key: str,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    db: TopologicalInterpDB,
    layer_descriptions: dict[str, str],
    k: int,
) -> list[NeighborContext]:
    """Top-K downstream (later-layer) components by absolute attribution."""
    source_layer, _ = parse_component_key(component_key)

    pos_targets = attribution_storage.get_top_component_targets(
        component_key, k=k * 2, sign="positive"
    )
    neg_targets = attribution_storage.get_top_component_targets(
        component_key, k=k * 2, sign="negative"
    )

    all_targets = pos_targets + neg_targets
    all_targets.sort(key=lambda e: abs(e.value), reverse=True)

    downstream = [
        e
        for e in all_targets
        if e.layer in layer_descriptions
        and is_later_layer(source_layer, e.layer, layer_descriptions)
    ]
    downstream = downstream[:k]

    cofiring = _build_cofiring_lookup(component_key, correlation_storage, k * 3)

    return [_build_neighbor_context(e.component_key, e.value, cofiring, db) for e in downstream]


def get_upstream_neighbors(
    component_key: str,
    attribution_storage: DatasetAttributionStorage,
    correlation_storage: CorrelationStorage,
    db: TopologicalInterpDB,
    layer_descriptions: dict[str, str],
    k: int,
) -> list[NeighborContext]:
    """Top-K upstream (earlier-layer) components by absolute attribution."""
    target_layer, _ = parse_component_key(component_key)

    pos_sources = attribution_storage.get_top_sources(component_key, k=k * 2, sign="positive")
    neg_sources = attribution_storage.get_top_sources(component_key, k=k * 2, sign="negative")

    all_sources = [e for e in pos_sources + neg_sources if not e.component_key.startswith("wte:")]
    all_sources.sort(key=lambda e: abs(e.value), reverse=True)

    upstream = [
        e
        for e in all_sources
        if e.layer in layer_descriptions
        and is_later_layer(e.layer, target_layer, layer_descriptions)
    ]
    upstream = upstream[:k]

    cofiring = _build_cofiring_lookup(component_key, correlation_storage, k * 3)

    return [_build_neighbor_context(e.component_key, e.value, cofiring, db) for e in upstream]


def get_cofiring_neighbors(
    component_key: str,
    correlation_storage: CorrelationStorage,
    k: int,
) -> list[NeighborContext]:
    """Top-K co-firing components by Jaccard similarity."""
    correlated = get_correlated_components(
        correlation_storage, component_key, metric="jaccard", top_k=k
    )

    pmi_lookup: dict[str, float] = {}
    pmi_results = get_correlated_components(
        correlation_storage, component_key, metric="pmi", top_k=k * 3
    )
    for c in pmi_results:
        pmi_lookup[c.component_key] = c.score

    return [
        NeighborContext(
            component_key=c.component_key,
            attribution=0.0,
            label=None,
            confidence=None,
            jaccard=c.score,
            pmi=pmi_lookup.get(c.component_key),
        )
        for c in correlated
    ]


def _build_cofiring_lookup(
    component_key: str,
    correlation_storage: CorrelationStorage,
    k: int,
) -> dict[str, tuple[float, float | None]]:
    """Build {neighbor_key: (jaccard, pmi)} lookup for co-firing stats."""
    lookup: dict[str, tuple[float, float | None]] = {}

    jaccard_results = get_correlated_components(
        correlation_storage, component_key, metric="jaccard", top_k=k
    )
    for c in jaccard_results:
        lookup[c.component_key] = (c.score, None)

    pmi_results = get_correlated_components(
        correlation_storage, component_key, metric="pmi", top_k=k
    )
    for c in pmi_results:
        if c.component_key in lookup:
            jaccard_val = lookup[c.component_key][0]
            lookup[c.component_key] = (jaccard_val, c.score)
        else:
            lookup[c.component_key] = (0.0, c.score)

    return lookup


def _build_neighbor_context(
    neighbor_key: str,
    attribution: float,
    cofiring: dict[str, tuple[float, float | None]],
    db: TopologicalInterpDB,
) -> NeighborContext:
    output_label = db.get_output_label(neighbor_key)
    jaccard, pmi = cofiring.get(neighbor_key, (None, None))

    return NeighborContext(
        component_key=neighbor_key,
        attribution=attribution,
        label=output_label.label if output_label else None,
        confidence=output_label.confidence if output_label else None,
        jaccard=jaccard,
        pmi=pmi,
    )
