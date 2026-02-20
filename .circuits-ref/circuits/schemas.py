"""Common data types for the circuits pipeline.

Defines Unit, Edge, and Graph schemas used across all stages:
graph generation, aggregation, labeling, clustering, and database building.

Usage:
    from circuits.schemas import Unit, Edge, Graph, UnitType, LabelSource

    unit = Unit(layer=24, index=5326, label="COX enzyme detector")
    edge = Edge(src_layer=20, src_index=455, tgt_layer=24, tgt_index=5326, weight=3.14)
    graph = Graph(units=[unit], edges=[edge], prompt="What is aspirin?")

    # Convert from/to legacy Neuronpedia format
    graph = Graph.from_legacy(json.load(open("graph.json")))
    legacy = graph.to_legacy()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UnitType(str, Enum):
    """Type of computational unit in the network."""
    NEURON = "neuron"
    TRANSCODER = "transcoder"
    ATTENTION_HEAD = "attention_head"


class LabelSource(str, Enum):
    """Provenance of a unit label."""
    NEURONDB = "neurondb"
    AUTOINTERP = "autointerp"
    PROGRESSIVE_OUTPUT = "progressive_output"
    PROGRESSIVE_INPUT = "progressive_input"
    HUMAN = "human"
    GOODFIRE = "goodfire"


class ConnectivityMethod(str, Enum):
    """How edges were computed."""
    RELP = "relp"
    OUTPUT_PROJECTION = "output_projection"
    WEIGHT_GRAPH = "weight_graph"


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnitLabel:
    """A single label with provenance tracking."""
    text: str
    source: LabelSource
    confidence: str = "unknown"  # low / medium / high
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source.value,
            "confidence": self.confidence,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnitLabel:
        return cls(
            text=d["text"],
            source=LabelSource(d["source"]),
            confidence=d.get("confidence", "unknown"),
            description=d.get("description", ""),
        )


@dataclass
class TokenProjection:
    """A token with its logit contribution weight."""
    token: str
    token_id: int
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"token": self.token, "token_id": self.token_id, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenProjection:
        return cls(token=d["token"], token_id=d["token_id"], weight=d["weight"])


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """A computational unit (neuron, transcoder feature, attention head).

    Holds identity, labels, activation stats, output projections, and
    cluster assignments.  Designed to be the single representation used
    across graph generation, aggregation, labeling, and database stages.
    """

    layer: int
    index: int
    unit_type: UnitType = UnitType.NEURON

    # Primary labels (best-available short strings)
    label: str = ""
    input_label: str = ""
    output_label: str = ""

    # All labels with provenance
    labels: list[UnitLabel] = field(default_factory=list)

    # Activation stats (populated during aggregation)
    max_activation: float | None = None
    appearance_count: int = 0

    # Output projections: context-independent logit effects
    # {"promotes": [TokenProjection, ...], "suppresses": [TokenProjection, ...]}
    output_projections: dict[str, list[TokenProjection]] | None = None

    # Input projections: what tokens excite this unit
    input_projections: list[TokenProjection] | None = None

    # Cluster assignment (from Infomap)
    cluster_path: str | None = None   # e.g. "1.5.3"
    top_cluster: int | None = None
    sub_module: int | None = None
    subsub_module: int | None = None

    # Misc metadata
    output_norm: float | None = None
    infomap_flow: float | None = None

    # Per-graph context (only meaningful for single-graph Units)
    position: int | None = None       # ctx_idx / token position
    activation: float | None = None   # activation value in this graph
    influence: float | None = None    # influence score in this graph

    # ---- properties ----

    @property
    def unit_id(self) -> str:
        """Human-readable identifier, e.g. 'L24/N5326'."""
        return f"L{self.layer}/N{self.index}"

    @property
    def db_key(self) -> tuple[int, int]:
        """Database primary key tuple."""
        return (self.layer, self.index)

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d: dict[str, Any] = {
            "layer": self.layer,
            "index": self.index,
            "unit_type": self.unit_type.value,
            "label": self.label,
            "input_label": self.input_label,
            "output_label": self.output_label,
            "labels": [l.to_dict() for l in self.labels],
            "max_activation": self.max_activation,
            "appearance_count": self.appearance_count,
            "cluster_path": self.cluster_path,
            "top_cluster": self.top_cluster,
            "sub_module": self.sub_module,
            "subsub_module": self.subsub_module,
            "output_norm": self.output_norm,
            "infomap_flow": self.infomap_flow,
        }
        if self.output_projections is not None:
            d["output_projections"] = {
                k: [tp.to_dict() for tp in v]
                for k, v in self.output_projections.items()
            }
        if self.input_projections is not None:
            d["input_projections"] = [tp.to_dict() for tp in self.input_projections]
        if self.position is not None:
            d["position"] = self.position
        if self.activation is not None:
            d["activation"] = self.activation
        if self.influence is not None:
            d["influence"] = self.influence
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unit:
        """Deserialize from a plain dict."""
        output_proj = None
        if d.get("output_projections"):
            output_proj = {
                k: [TokenProjection.from_dict(tp) for tp in v]
                for k, v in d["output_projections"].items()
            }
        input_proj = None
        if d.get("input_projections"):
            input_proj = [TokenProjection.from_dict(tp) for tp in d["input_projections"]]
        labels = [UnitLabel.from_dict(l) for l in d.get("labels", [])]
        return cls(
            layer=d["layer"],
            index=d["index"],
            unit_type=UnitType(d.get("unit_type", "neuron")),
            label=d.get("label", ""),
            input_label=d.get("input_label", ""),
            output_label=d.get("output_label", ""),
            labels=labels,
            max_activation=d.get("max_activation"),
            appearance_count=d.get("appearance_count", 0),
            output_projections=output_proj,
            input_projections=input_proj,
            cluster_path=d.get("cluster_path"),
            top_cluster=d.get("top_cluster"),
            sub_module=d.get("sub_module"),
            subsub_module=d.get("subsub_module"),
            output_norm=d.get("output_norm"),
            infomap_flow=d.get("infomap_flow"),
            position=d.get("position"),
            activation=d.get("activation"),
            influence=d.get("influence"),
        )

    # ---- legacy conversion ----

    @classmethod
    def from_graph_node(cls, node: dict) -> Unit | None:
        """Convert a legacy graph node dict to a Unit.

        Returns None for embedding or logit nodes (non-neuron layers).
        """
        layer = node.get("layer", "")
        feature_type = node.get("feature_type", "")

        # Skip embeddings and logits
        if layer == "E" or feature_type == "embedding":
            return None
        if node.get("isLogit", False) or feature_type == "logit":
            return None

        # Only accept integer layers
        try:
            layer_int = int(layer)
        except (ValueError, TypeError):
            return None

        feature = node.get("feature", node.get("neuron", 0))
        clerp = node.get("clerp", "")

        return cls(
            layer=layer_int,
            index=feature,
            label=clerp,
            position=node.get("ctx_idx", node.get("position")),
            activation=node.get("activation"),
            influence=node.get("influence"),
        )

    def to_graph_node(self, node_id: str | None = None) -> dict[str, Any]:
        """Convert back to legacy graph node dict."""
        nid = node_id or f"{self.layer}_{self.index}_{self.position or 0}"
        return {
            "node_id": nid,
            "feature": self.index,
            "layer": str(self.layer),
            "ctx_idx": self.position or 0,
            "feature_type": "mlp_neuron",
            "jsNodeId": nid,
            "clerp": self.label or f"L{self.layer}/N{self.index}",
            "influence": self.influence,
            "activation": self.activation,
            "isLogit": False,
        }


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """A directed edge between two units.

    For single-graph edges, ``weight`` is the attribution weight.
    For aggregated edges, the running statistics (count, weight_sum, etc.)
    track the distribution across many graphs.
    """

    src_layer: int
    src_index: int
    tgt_layer: int
    tgt_index: int

    weight: float = 0.0           # Single-graph weight OR best available
    count: int = 1                # Number of graphs containing this edge
    weight_sum: float = 0.0
    weight_abs_sum: float = 0.0
    weight_sq_sum: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    method: ConnectivityMethod = ConnectivityMethod.RELP

    # ---- computed properties ----

    @property
    def mean_weight(self) -> float:
        if self.count == 0:
            return 0.0
        return self.weight_sum / self.count

    @property
    def mean_abs_weight(self) -> float:
        if self.count == 0:
            return 0.0
        return self.weight_abs_sum / self.count

    @property
    def std_weight(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean_weight
        variance = (self.weight_sq_sum / self.count) - (mean * mean)
        return max(variance, 0.0) ** 0.5

    @property
    def src_key(self) -> tuple[int, int]:
        return (self.src_layer, self.src_index)

    @property
    def tgt_key(self) -> tuple[int, int]:
        return (self.tgt_layer, self.tgt_index)

    @property
    def edge_key(self) -> tuple[int, int, int, int]:
        return (self.src_layer, self.src_index, self.tgt_layer, self.tgt_index)

    # ---- mutation ----

    def accumulate(self, weight: float) -> None:
        """Add an observation (for aggregation)."""
        self.count += 1
        self.weight_sum += weight
        self.weight_abs_sum += abs(weight)
        self.weight_sq_sum += weight * weight
        if weight < self.weight_min:
            self.weight_min = weight
        if weight > self.weight_max:
            self.weight_max = weight

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_layer": self.src_layer,
            "src_index": self.src_index,
            "tgt_layer": self.tgt_layer,
            "tgt_index": self.tgt_index,
            "weight": self.weight,
            "count": self.count,
            "weight_sum": self.weight_sum,
            "weight_abs_sum": self.weight_abs_sum,
            "weight_sq_sum": self.weight_sq_sum,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "method": self.method.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        return cls(
            src_layer=d["src_layer"],
            src_index=d["src_index"],
            tgt_layer=d["tgt_layer"],
            tgt_index=d["tgt_index"],
            weight=d.get("weight", 0.0),
            count=d.get("count", 1),
            weight_sum=d.get("weight_sum", 0.0),
            weight_abs_sum=d.get("weight_abs_sum", 0.0),
            weight_sq_sum=d.get("weight_sq_sum", 0.0),
            weight_min=d.get("weight_min", 0.0),
            weight_max=d.get("weight_max", 0.0),
            method=ConnectivityMethod(d.get("method", "relp")),
        )

    # ---- legacy conversion ----

    @classmethod
    def from_graph_link(cls, link: dict) -> Edge | None:
        """Convert a legacy graph link dict to an Edge.

        Legacy format: {"source": "1_2427_0", "target": "2_512_3", "weight": 6.34}
        Node IDs are "{layer}_{feature}_{position}".
        """
        source = link.get("source", "")
        target = link.get("target", "")
        weight = link.get("weight", 0.0)

        src_parts = source.split("_")
        tgt_parts = target.split("_")

        if len(src_parts) < 2 or len(tgt_parts) < 2:
            return None

        try:
            src_layer = int(src_parts[0])
            src_index = int(src_parts[1])
            tgt_layer = int(tgt_parts[0])
            tgt_index = int(tgt_parts[1])
        except ValueError:
            # Embedding (E_*) or Logit (L_*) links
            return None

        return cls(
            src_layer=src_layer,
            src_index=src_index,
            tgt_layer=tgt_layer,
            tgt_index=tgt_index,
            weight=weight,
            weight_sum=weight,
            weight_abs_sum=abs(weight),
            weight_sq_sum=weight * weight,
            weight_min=weight,
            weight_max=weight,
        )

    def to_graph_link(self, src_position: int = 0, tgt_position: int = 0) -> dict[str, Any]:
        """Convert back to legacy graph link dict."""
        return {
            "source": f"{self.src_layer}_{self.src_index}_{src_position}",
            "target": f"{self.tgt_layer}_{self.tgt_index}_{tgt_position}",
            "weight": self.weight,
        }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    """A complete attribution graph with units, edges, and metadata.

    Wraps the legacy dict-based format with typed access while supporting
    lossless round-trip conversion via from_legacy / to_legacy.
    """

    units: list[Unit] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    prompt_tokens: list[str] = field(default_factory=list)

    # Legacy fields preserved for round-trip fidelity
    _raw_nodes: list[dict[str, Any]] | None = field(
        default=None, repr=False
    )
    _raw_links: list[dict[str, Any]] | None = field(
        default=None, repr=False
    )

    # ---- properties ----

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def get_unit(self, layer: int, index: int) -> Unit | None:
        """Look up a unit by layer and index."""
        for u in self.units:
            if u.layer == layer and u.index == index:
                return u
        return None

    def unit_lookup(self) -> dict[tuple[int, int], Unit]:
        """Build a {(layer, index): Unit} mapping."""
        return {u.db_key: u for u in self.units}

    # ---- legacy conversion ----

    @classmethod
    def from_legacy(cls, graph_data: dict) -> Graph:
        """Convert a legacy Neuronpedia-format graph dict to a Graph.

        Preserves raw nodes/links for lossless round-trip via to_legacy().
        """
        metadata = dict(graph_data.get("metadata", {}))
        prompt = metadata.get("prompt", "")
        prompt_tokens = metadata.get("prompt_tokens", [])

        # Parse units (neurons only, skip embeddings/logits)
        units = []
        for node in graph_data.get("nodes", []):
            unit = Unit.from_graph_node(node)
            if unit is not None:
                units.append(unit)

        # Parse edges (skip embedding/logit edges)
        edges = []
        for link in graph_data.get("links", []):
            edge = Edge.from_graph_link(link)
            if edge is not None:
                edges.append(edge)

        return cls(
            units=units,
            edges=edges,
            metadata=metadata,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            _raw_nodes=graph_data.get("nodes"),
            _raw_links=graph_data.get("links"),
        )

    def to_legacy(self) -> dict:
        """Convert back to legacy Neuronpedia-format dict.

        If raw nodes/links were preserved from from_legacy(), uses those
        for perfect round-trip fidelity.  Otherwise reconstructs from
        typed units/edges.
        """
        if self._raw_nodes is not None and self._raw_links is not None:
            # Perfect round-trip: use original data
            metadata = dict(self.metadata)
            metadata["prompt"] = self.prompt
            metadata["prompt_tokens"] = self.prompt_tokens
            return {
                "metadata": metadata,
                "nodes": self._raw_nodes,
                "links": self._raw_links,
                "qParams": {"pinnedIds": [], "supernodes": [], "linkType": "both"},
                "features": _build_features(self._raw_nodes),
            }

        # Build position lookup for link IDs
        pos_lookup: dict[tuple[int, int], int] = {}
        for u in self.units:
            pos_lookup[(u.layer, u.index)] = u.position or 0

        nodes = [u.to_graph_node() for u in self.units]
        links = [
            e.to_graph_link(
                src_position=pos_lookup.get((e.src_layer, e.src_index), 0),
                tgt_position=pos_lookup.get((e.tgt_layer, e.tgt_index), 0),
            )
            for e in self.edges
        ]

        metadata = dict(self.metadata)
        metadata["prompt"] = self.prompt
        metadata["prompt_tokens"] = self.prompt_tokens

        return {
            "metadata": metadata,
            "nodes": nodes,
            "links": links,
            "qParams": {"pinnedIds": [], "supernodes": [], "linkType": "both"},
            "features": _build_features(nodes),
        }

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict using the new schema format."""
        return {
            "units": [u.to_dict() for u in self.units],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Graph:
        """Deserialize from the new schema format."""
        return cls(
            units=[Unit.from_dict(u) for u in d.get("units", [])],
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
            metadata=d.get("metadata", {}),
            prompt=d.get("prompt", ""),
            prompt_tokens=d.get("prompt_tokens", []),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_features(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the legacy 'features' list from nodes."""
    features = []
    seen = set()
    for node in nodes:
        fid = f"{node.get('layer', '')}_{node.get('feature', '')}"
        if fid not in seen:
            seen.add(fid)
            features.append({
                "featureId": fid,
                "featureIndex": node.get("feature"),
                "layer": node.get("layer"),
                "clerp": node.get("clerp", ""),
                "feature_type": node.get("feature_type", ""),
            })
    return features
