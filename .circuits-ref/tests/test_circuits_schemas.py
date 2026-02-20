"""Tests for the circuits package: schemas, aggregation, database, and CLI.

All tests run WITHOUT a GPU or model. Uses tmp_path for file I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from circuits.aggregation import InMemoryAggregator
from circuits.database import CircuitDatabase
from circuits.schemas import (
    ConnectivityMethod,
    Edge,
    Graph,
    LabelSource,
    TokenProjection,
    Unit,
    UnitLabel,
    UnitType,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_LEGACY_GRAPH = {
    "metadata": {
        "prompt": "The capital of France is",
        "prompt_tokens": ["The", " capital", " of", " France", " is"],
        "node_threshold": 0.005,
    },
    "nodes": [
        {
            "node_id": "E_128000_0",
            "feature": 128000,
            "layer": "E",
            "ctx_idx": 0,
            "feature_type": "embedding",
            "jsNodeId": "E_128000_0",
            "clerp": "Token: <bos>",
            "influence": None,
            "activation": 1.0,
            "isLogit": False,
        },
        {
            "node_id": "10_2427_3",
            "feature": 2427,
            "layer": 10,
            "ctx_idx": 3,
            "feature_type": "mlp_neuron",
            "jsNodeId": "10_2427_3",
            "clerp": "L10/N2427: geography",
            "influence": 2.5,
            "activation": 0.8,
            "isLogit": False,
        },
        {
            "node_id": "24_5326_4",
            "feature": 5326,
            "layer": 24,
            "ctx_idx": 4,
            "feature_type": "mlp_neuron",
            "jsNodeId": "24_5326_4",
            "clerp": "L24/N5326: capital cities",
            "influence": 4.1,
            "activation": 1.2,
            "isLogit": False,
        },
        {
            "node_id": "L_3681_4",
            "feature": 3681,
            "layer": 32,
            "ctx_idx": 4,
            "feature_type": "logit",
            "jsNodeId": "L_3681_4",
            "clerp": "Logit: Paris",
            "influence": None,
            "activation": None,
            "isLogit": True,
        },
    ],
    "links": [
        {"source": "E_128000_0", "target": "10_2427_3", "weight": 1.5},
        {"source": "10_2427_3", "target": "24_5326_4", "weight": 3.14},
        {"source": "24_5326_4", "target": "L_3681_4", "weight": 5.0},
    ],
    "qParams": {"pinnedIds": [], "supernodes": [], "linkType": "both"},
}


def _make_graph_json(nodes, links, prompt="test", corpus_index=0):
    """Build a minimal legacy graph dict for test graph files."""
    return {
        "metadata": {
            "prompt": prompt,
            "prompt_tokens": prompt.split(),
            "corpus_index": corpus_index,
        },
        "nodes": nodes,
        "links": links,
    }


# =========================================================================
# 1. circuits.schemas -- Unit
# =========================================================================


class TestUnit:
    """Unit dataclass tests."""

    def test_minimal_construction(self):
        u = Unit(layer=24, index=5326)
        assert u.layer == 24
        assert u.index == 5326
        assert u.unit_type == UnitType.NEURON
        assert u.label == ""
        assert u.appearance_count == 0
        assert u.max_activation is None

    def test_unit_id_property(self):
        u = Unit(layer=24, index=5326)
        assert u.unit_id == "L24/N5326"

    def test_unit_id_layer_zero(self):
        u = Unit(layer=0, index=42)
        assert u.unit_id == "L0/N42"

    def test_db_key_property(self):
        u = Unit(layer=24, index=5326)
        assert u.db_key == (24, 5326)

    def test_to_dict_from_dict_roundtrip_minimal(self):
        u = Unit(layer=10, index=999)
        d = u.to_dict()
        u2 = Unit.from_dict(d)
        assert u2.layer == u.layer
        assert u2.index == u.index
        assert u2.unit_type == u.unit_type
        assert u2.label == u.label
        assert u2.appearance_count == u.appearance_count

    def test_to_dict_from_dict_roundtrip_full(self):
        u = Unit(
            layer=24,
            index=5326,
            unit_type=UnitType.NEURON,
            label="capital cities",
            input_label="France-related tokens",
            output_label="promotes Paris",
            labels=[
                UnitLabel(
                    text="capital cities",
                    source=LabelSource.AUTOINTERP,
                    confidence="high",
                    description="Detects capital city references",
                )
            ],
            max_activation=12.5,
            appearance_count=42,
            cluster_path="1.5.3",
            top_cluster=1,
            sub_module=5,
            subsub_module=3,
            output_norm=0.75,
            infomap_flow=0.002,
            position=4,
            activation=1.2,
            influence=4.1,
        )
        d = u.to_dict()
        u2 = Unit.from_dict(d)

        assert u2.layer == 24
        assert u2.index == 5326
        assert u2.label == "capital cities"
        assert u2.input_label == "France-related tokens"
        assert u2.output_label == "promotes Paris"
        assert len(u2.labels) == 1
        assert u2.labels[0].text == "capital cities"
        assert u2.labels[0].source == LabelSource.AUTOINTERP
        assert u2.labels[0].confidence == "high"
        assert u2.labels[0].description == "Detects capital city references"
        assert u2.max_activation == 12.5
        assert u2.appearance_count == 42
        assert u2.cluster_path == "1.5.3"
        assert u2.top_cluster == 1
        assert u2.sub_module == 5
        assert u2.subsub_module == 3
        assert u2.output_norm == 0.75
        assert u2.infomap_flow == 0.002
        assert u2.position == 4
        assert u2.activation == 1.2
        assert u2.influence == 4.1

    def test_to_dict_with_output_projections(self):
        u = Unit(
            layer=15,
            index=100,
            output_projections={
                "promotes": [
                    TokenProjection(token="Paris", token_id=3681, weight=2.5),
                    TokenProjection(token="London", token_id=4100, weight=1.1),
                ],
                "suppresses": [
                    TokenProjection(token="dog", token_id=999, weight=-0.8),
                ],
            },
        )
        d = u.to_dict()
        assert "output_projections" in d
        assert len(d["output_projections"]["promotes"]) == 2
        assert d["output_projections"]["promotes"][0]["token"] == "Paris"
        assert d["output_projections"]["suppresses"][0]["weight"] == -0.8

        u2 = Unit.from_dict(d)
        assert len(u2.output_projections["promotes"]) == 2
        assert u2.output_projections["promotes"][0].token == "Paris"
        assert u2.output_projections["promotes"][0].token_id == 3681

    def test_to_dict_with_labels_list(self):
        u = Unit(
            layer=5,
            index=200,
            labels=[
                UnitLabel(text="geography", source=LabelSource.NEURONDB),
                UnitLabel(
                    text="European capitals",
                    source=LabelSource.AUTOINTERP,
                    confidence="medium",
                ),
            ],
        )
        d = u.to_dict()
        assert len(d["labels"]) == 2
        assert d["labels"][0]["source"] == "neurondb"
        assert d["labels"][1]["source"] == "autointerp"

    def test_from_graph_node_mlp_neuron(self):
        node = {
            "node_id": "10_2427_3",
            "feature": 2427,
            "layer": 10,
            "ctx_idx": 3,
            "feature_type": "mlp_neuron",
            "clerp": "L10/N2427: geography",
            "influence": 2.5,
            "activation": 0.8,
            "isLogit": False,
        }
        u = Unit.from_graph_node(node)
        assert u is not None
        assert u.layer == 10
        assert u.index == 2427
        assert u.position == 3
        assert u.label == "L10/N2427: geography"
        assert u.influence == 2.5
        assert u.activation == 0.8

    def test_from_graph_node_returns_none_for_embedding(self):
        node = {
            "node_id": "E_128000_0",
            "feature": 128000,
            "layer": "E",
            "ctx_idx": 0,
            "feature_type": "embedding",
            "isLogit": False,
        }
        assert Unit.from_graph_node(node) is None

    def test_from_graph_node_returns_none_for_logit(self):
        node = {
            "node_id": "L_3681_4",
            "feature": 3681,
            "layer": 32,
            "ctx_idx": 4,
            "feature_type": "logit",
            "isLogit": True,
        }
        assert Unit.from_graph_node(node) is None

    def test_from_graph_node_returns_none_for_logit_by_feature_type(self):
        node = {
            "node_id": "L_100_0",
            "feature": 100,
            "layer": 32,
            "feature_type": "logit",
            "isLogit": False,
        }
        assert Unit.from_graph_node(node) is None

    def test_to_graph_node_produces_valid_legacy(self):
        u = Unit(
            layer=24,
            index=5326,
            label="capital cities",
            position=4,
            activation=1.2,
            influence=4.1,
        )
        node = u.to_graph_node()
        assert node["feature"] == 5326
        assert node["layer"] == "24"
        assert node["ctx_idx"] == 4
        assert node["feature_type"] == "mlp_neuron"
        assert node["isLogit"] is False
        assert node["clerp"] == "capital cities"
        assert node["influence"] == 4.1
        assert node["activation"] == 1.2
        assert node["node_id"] == "24_5326_4"
        assert node["jsNodeId"] == "24_5326_4"

    def test_to_graph_node_no_label_fallback(self):
        u = Unit(layer=10, index=500)
        node = u.to_graph_node()
        assert node["clerp"] == "L10/N500"

    def test_to_graph_node_custom_node_id(self):
        u = Unit(layer=10, index=500, position=2)
        node = u.to_graph_node(node_id="custom_id")
        assert node["node_id"] == "custom_id"
        assert node["jsNodeId"] == "custom_id"


# =========================================================================
# 1. circuits.schemas -- Edge
# =========================================================================


class TestEdge:
    """Edge dataclass tests."""

    def test_construction_and_properties(self):
        e = Edge(
            src_layer=10,
            src_index=2427,
            tgt_layer=24,
            tgt_index=5326,
            weight=3.14,
            count=2,
            weight_sum=6.28,
            weight_abs_sum=6.28,
            weight_sq_sum=19.72,
        )
        assert e.mean_weight == pytest.approx(3.14)
        assert e.mean_abs_weight == pytest.approx(3.14)

    def test_mean_weight_zero_count(self):
        e = Edge(src_layer=0, src_index=0, tgt_layer=1, tgt_index=1, count=0)
        assert e.mean_weight == 0.0
        assert e.mean_abs_weight == 0.0

    def test_std_weight(self):
        # Two observations: 2.0 and 4.0 => mean=3.0, var=1.0, std=1.0
        e = Edge(
            src_layer=0,
            src_index=0,
            tgt_layer=1,
            tgt_index=1,
            count=2,
            weight_sum=6.0,
            weight_sq_sum=20.0,  # 4 + 16
        )
        assert e.std_weight == pytest.approx(1.0)

    def test_std_weight_single_observation(self):
        e = Edge(
            src_layer=0,
            src_index=0,
            tgt_layer=1,
            tgt_index=1,
            count=1,
            weight_sum=5.0,
            weight_sq_sum=25.0,
        )
        assert e.std_weight == 0.0

    def test_accumulate_three_observations(self):
        e = Edge(
            src_layer=10,
            src_index=100,
            tgt_layer=20,
            tgt_index=200,
            weight=0.0,
            count=0,
            weight_sum=0.0,
            weight_abs_sum=0.0,
            weight_sq_sum=0.0,
            weight_min=0.0,
            weight_max=0.0,
        )

        e.accumulate(3.0)
        assert e.count == 1
        assert e.weight_sum == pytest.approx(3.0)
        assert e.weight_abs_sum == pytest.approx(3.0)
        assert e.weight_sq_sum == pytest.approx(9.0)
        assert e.weight_max == pytest.approx(3.0)

        e.accumulate(-1.0)
        assert e.count == 2
        assert e.weight_sum == pytest.approx(2.0)
        assert e.weight_abs_sum == pytest.approx(4.0)
        assert e.weight_sq_sum == pytest.approx(10.0)
        assert e.weight_min == pytest.approx(-1.0)
        assert e.weight_max == pytest.approx(3.0)

        e.accumulate(5.0)
        assert e.count == 3
        assert e.weight_sum == pytest.approx(7.0)
        assert e.weight_abs_sum == pytest.approx(9.0)
        assert e.weight_sq_sum == pytest.approx(35.0)
        assert e.weight_min == pytest.approx(-1.0)
        assert e.weight_max == pytest.approx(5.0)

        assert e.mean_weight == pytest.approx(7.0 / 3)
        assert e.mean_abs_weight == pytest.approx(9.0 / 3)

    def test_to_dict_from_dict_roundtrip(self):
        e = Edge(
            src_layer=10,
            src_index=2427,
            tgt_layer=24,
            tgt_index=5326,
            weight=3.14,
            count=5,
            weight_sum=15.7,
            weight_abs_sum=15.7,
            weight_sq_sum=49.3,
            weight_min=2.0,
            weight_max=4.0,
            method=ConnectivityMethod.RELP,
        )
        d = e.to_dict()
        e2 = Edge.from_dict(d)
        assert e2.src_layer == 10
        assert e2.src_index == 2427
        assert e2.tgt_layer == 24
        assert e2.tgt_index == 5326
        assert e2.weight == pytest.approx(3.14)
        assert e2.count == 5
        assert e2.weight_sum == pytest.approx(15.7)
        assert e2.weight_abs_sum == pytest.approx(15.7)
        assert e2.weight_sq_sum == pytest.approx(49.3)
        assert e2.weight_min == pytest.approx(2.0)
        assert e2.weight_max == pytest.approx(4.0)
        assert e2.method == ConnectivityMethod.RELP

    def test_from_graph_link_neuron_to_neuron(self):
        link = {"source": "10_2427_3", "target": "24_5326_4", "weight": 3.14}
        e = Edge.from_graph_link(link)
        assert e is not None
        assert e.src_layer == 10
        assert e.src_index == 2427
        assert e.tgt_layer == 24
        assert e.tgt_index == 5326
        assert e.weight == pytest.approx(3.14)
        assert e.weight_sum == pytest.approx(3.14)
        assert e.weight_abs_sum == pytest.approx(3.14)
        assert e.weight_sq_sum == pytest.approx(3.14 * 3.14)
        assert e.weight_min == pytest.approx(3.14)
        assert e.weight_max == pytest.approx(3.14)

    def test_from_graph_link_returns_none_for_embedding(self):
        link = {"source": "E_128000_0", "target": "10_2427_3", "weight": 1.5}
        assert Edge.from_graph_link(link) is None

    def test_from_graph_link_returns_none_for_logit_target(self):
        link = {"source": "24_5326_4", "target": "L_3681_4", "weight": 5.0}
        assert Edge.from_graph_link(link) is None

    def test_edge_key_src_key_tgt_key(self):
        e = Edge(src_layer=10, src_index=2427, tgt_layer=24, tgt_index=5326)
        assert e.edge_key == (10, 2427, 24, 5326)
        assert e.src_key == (10, 2427)
        assert e.tgt_key == (24, 5326)


# =========================================================================
# 1. circuits.schemas -- Graph
# =========================================================================


class TestGraph:
    """Graph dataclass tests."""

    def test_empty_graph(self):
        g = Graph()
        assert g.n_units == 0
        assert g.n_edges == 0
        assert g.prompt == ""
        assert g.prompt_tokens == []
        assert g.metadata == {}

    def test_from_legacy_filters_nodes(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        # 4 nodes total: 1 embedding, 2 neurons, 1 logit => only 2 units
        assert g.n_units == 2
        unit_ids = {u.unit_id for u in g.units}
        assert "L10/N2427" in unit_ids
        assert "L24/N5326" in unit_ids

    def test_from_legacy_filters_edges(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        # 3 links: E->10 (skip), 10->24 (keep), 24->L (skip) => 1 edge
        assert g.n_edges == 1
        assert g.edges[0].src_layer == 10
        assert g.edges[0].tgt_layer == 24

    def test_from_legacy_preserves_metadata(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        assert g.prompt == "The capital of France is"
        assert g.prompt_tokens == ["The", " capital", " of", " France", " is"]
        assert g.metadata["node_threshold"] == 0.005

    def test_to_legacy_roundtrip_preserves_raw(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        legacy = g.to_legacy()
        # Raw nodes/links preserved exactly
        assert len(legacy["nodes"]) == len(SAMPLE_LEGACY_GRAPH["nodes"])
        assert len(legacy["links"]) == len(SAMPLE_LEGACY_GRAPH["links"])
        assert legacy["metadata"]["prompt"] == "The capital of France is"

    def test_to_dict_from_dict_roundtrip(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        d = g.to_dict()
        g2 = Graph.from_dict(d)
        assert g2.n_units == g.n_units
        assert g2.n_edges == g.n_edges
        assert g2.prompt == g.prompt
        assert g2.prompt_tokens == g.prompt_tokens
        assert g2.units[0].layer == g.units[0].layer
        assert g2.units[0].index == g.units[0].index
        assert g2.edges[0].src_layer == g.edges[0].src_layer
        assert g2.edges[0].weight == pytest.approx(g.edges[0].weight)

    def test_get_unit_found(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        u = g.get_unit(24, 5326)
        assert u is not None
        assert u.unit_id == "L24/N5326"

    def test_get_unit_not_found(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        assert g.get_unit(99, 9999) is None

    def test_unit_lookup(self):
        g = Graph.from_legacy(SAMPLE_LEGACY_GRAPH)
        lookup = g.unit_lookup()
        assert (10, 2427) in lookup
        assert (24, 5326) in lookup
        assert lookup[(24, 5326)].label == "L24/N5326: capital cities"
        assert len(lookup) == 2

    def test_n_units_and_n_edges(self):
        g = Graph(
            units=[Unit(layer=0, index=1), Unit(layer=0, index=2), Unit(layer=1, index=3)],
            edges=[
                Edge(src_layer=0, src_index=1, tgt_layer=1, tgt_index=3, weight=1.0),
            ],
        )
        assert g.n_units == 3
        assert g.n_edges == 1


# =========================================================================
# 2. circuits.aggregation -- InMemoryAggregator
# =========================================================================


def _write_test_graphs(tmp_path: Path):
    """Write two small graph JSON files and return their paths."""
    graph1 = _make_graph_json(
        nodes=[
            {
                "node_id": "5_100_0",
                "feature": 100,
                "layer": 5,
                "ctx_idx": 0,
                "feature_type": "mlp_neuron",
                "isLogit": False,
            },
            {
                "node_id": "10_200_1",
                "feature": 200,
                "layer": 10,
                "ctx_idx": 1,
                "feature_type": "mlp_neuron",
                "isLogit": False,
            },
            {
                "node_id": "E_128000_0",
                "feature": 128000,
                "layer": "E",
                "feature_type": "embedding",
                "isLogit": False,
            },
        ],
        links=[
            {"source": "5_100_0", "target": "10_200_1", "weight": 2.0},
            {"source": "E_128000_0", "target": "5_100_0", "weight": 1.0},
        ],
        prompt="test one",
        corpus_index=0,
    )

    graph2 = _make_graph_json(
        nodes=[
            {
                "node_id": "5_100_0",
                "feature": 100,
                "layer": 5,
                "ctx_idx": 0,
                "feature_type": "mlp_neuron",
                "isLogit": False,
            },
            {
                "node_id": "10_200_1",
                "feature": 200,
                "layer": 10,
                "ctx_idx": 1,
                "feature_type": "mlp_neuron",
                "isLogit": False,
            },
            {
                "node_id": "15_300_2",
                "feature": 300,
                "layer": 15,
                "ctx_idx": 2,
                "feature_type": "mlp_neuron",
                "isLogit": False,
            },
        ],
        links=[
            {"source": "5_100_0", "target": "10_200_1", "weight": 4.0},
            {"source": "10_200_1", "target": "15_300_2", "weight": -1.5},
        ],
        prompt="test two",
        corpus_index=1,
    )

    p1 = tmp_path / "graph_001.json"
    p2 = tmp_path / "graph_002.json"
    p1.write_text(json.dumps(graph1))
    p2.write_text(json.dumps(graph2))
    return p1, p2


class TestInMemoryAggregator:
    """InMemoryAggregator tests."""

    def test_process_two_graphs_edge_counts(self, tmp_path):
        p1, p2 = _write_test_graphs(tmp_path)
        agg = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg.process_graph(p1)
        agg.process_graph(p2)

        # edge 5_100 -> 10_200 appears in both graphs
        key = (5, 100, 10, 200)
        assert key in agg.edges
        stats = agg.edges[key]
        assert int(stats[0]) == 2  # count
        assert stats[1] == pytest.approx(6.0)  # sum (2+4)
        assert stats[2] == pytest.approx(6.0)  # abs_sum

        # edge 10_200 -> 15_300 only in graph2
        key2 = (10, 200, 15, 300)
        assert key2 in agg.edges
        assert int(agg.edges[key2][0]) == 1

        assert agg.graphs_processed == 2

    def test_get_edges_returns_edge_objects(self, tmp_path):
        p1, p2 = _write_test_graphs(tmp_path)
        agg = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg.process_graph(p1)
        agg.process_graph(p2)

        edges = agg.get_edges()
        assert len(edges) >= 2
        # Check that they are Edge schema objects
        for e in edges:
            assert isinstance(e, Edge)

        # Find the edge with count=2
        shared = [e for e in edges if e.edge_key == (5, 100, 10, 200)]
        assert len(shared) == 1
        assert shared[0].count == 2
        assert shared[0].mean_weight == pytest.approx(3.0)  # (2+4)/2

    def test_get_units_returns_unit_objects(self, tmp_path):
        p1, p2 = _write_test_graphs(tmp_path)
        agg = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg.process_graph(p1)
        agg.process_graph(p2)

        units = agg.get_units()
        assert len(units) >= 2
        for u in units:
            assert isinstance(u, Unit)

        # neuron (5, 100) appears in both graphs
        n5_100 = [u for u in units if u.layer == 5 and u.index == 100]
        assert len(n5_100) == 1
        assert n5_100[0].appearance_count == 2

        # neuron (15, 300) only in graph 2
        n15_300 = [u for u in units if u.layer == 15 and u.index == 300]
        assert len(n15_300) == 1
        assert n15_300[0].appearance_count == 1

    def test_get_edges_min_count_filter(self, tmp_path):
        p1, p2 = _write_test_graphs(tmp_path)
        agg = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg.process_graph(p1)
        agg.process_graph(p2)

        edges_all = agg.get_edges(min_count=1)
        edges_filtered = agg.get_edges(min_count=2)

        # Only the shared edge (5,100)->(10,200) has count=2
        assert len(edges_filtered) < len(edges_all)
        assert all(e.count >= 2 for e in edges_filtered)
        assert len(edges_filtered) == 1
        assert edges_filtered[0].edge_key == (5, 100, 10, 200)

    def test_checkpoint_resume(self, tmp_path):
        p1, p2 = _write_test_graphs(tmp_path)

        # Process graph 1, checkpoint
        agg1 = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg1.process_graph(p1)
        ckpt_path = tmp_path / "ckpt" / "checkpoint_1.dat"
        agg1.checkpoint(ckpt_path)

        # Create new aggregator, resume, process graph 2
        agg2 = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg2.resume(ckpt_path)
        agg2.process_graph(p2)

        # Also do both at once for comparison
        agg_both = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt2")
        agg_both.process_graph(p1)
        agg_both.process_graph(p2)

        # Verify counts match
        assert agg2.graphs_processed == agg_both.graphs_processed
        assert len(agg2.edges) == len(agg_both.edges)
        for key, stats_both in agg_both.edges.items():
            assert key in agg2.edges, f"Missing edge {key} after resume"
            stats_resumed = agg2.edges[key]
            assert int(stats_resumed[0]) == int(stats_both[0])  # count
            assert stats_resumed[1] == pytest.approx(stats_both[1])  # sum

    def test_process_graph_skips_embedding_and_logit_links(self, tmp_path):
        graph_with_embeds = _make_graph_json(
            nodes=[
                {
                    "node_id": "E_128000_0",
                    "feature": 128000,
                    "layer": "E",
                    "feature_type": "embedding",
                    "isLogit": False,
                },
                {
                    "node_id": "5_100_0",
                    "feature": 100,
                    "layer": 5,
                    "feature_type": "mlp_neuron",
                    "isLogit": False,
                },
                {
                    "node_id": "L_999_0",
                    "feature": 999,
                    "layer": 32,
                    "feature_type": "logit",
                    "isLogit": True,
                },
            ],
            links=[
                {"source": "E_128000_0", "target": "5_100_0", "weight": 1.0},
                {"source": "5_100_0", "target": "L_999_0", "weight": 2.0},
            ],
        )

        p = tmp_path / "graph_embed.json"
        p.write_text(json.dumps(graph_with_embeds))

        agg = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg.process_graph(p)

        # Both links should be skipped (E_* source, L_* target)
        assert len(agg.edges) == 0
        assert agg.graphs_processed == 1
        # Only the neuron node (5,100) should be tracked (E and L are skipped)
        assert (5, 100) in agg.neurons

    def test_process_directory(self, tmp_path):
        _write_test_graphs(tmp_path)
        agg = InMemoryAggregator(graph_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt")
        agg.process_directory(tmp_path, checkpoint_interval=9999)
        assert agg.graphs_processed == 2
        assert len(agg.edges) >= 2


# =========================================================================
# 3. circuits.database -- CircuitDatabase
# =========================================================================


class TestCircuitDatabase:
    """CircuitDatabase tests using in-memory-like DuckDB via tmp_path."""

    def _make_db(self, tmp_path: Path) -> CircuitDatabase:
        db_path = tmp_path / "test.duckdb"
        db = CircuitDatabase(db_path)
        db.create_tables()
        return db

    def test_create_and_write_units(self, tmp_path):
        db = self._make_db(tmp_path)
        units = [
            Unit(layer=10, index=100, label="geography"),
            Unit(layer=24, index=200, label="capital cities", appearance_count=5),
        ]
        count = db.write_units(units)
        assert count == 2

        u = db.get_unit(10, 100)
        assert u is not None
        assert u.layer == 10
        assert u.index == 100
        assert u.label == "geography"

        u2 = db.get_unit(24, 200)
        assert u2 is not None
        assert u2.appearance_count == 5
        db.close()

    def test_get_unit_not_found(self, tmp_path):
        db = self._make_db(tmp_path)
        assert db.get_unit(99, 999) is None
        db.close()

    def test_write_and_read_edges(self, tmp_path):
        db = self._make_db(tmp_path)

        # Write some units first (edges reference them conceptually)
        units = [
            Unit(layer=10, index=100, label="geo"),
            Unit(layer=24, index=200, label="capitals"),
        ]
        db.write_units(units)

        edges = [
            Edge(
                src_layer=10,
                src_index=100,
                tgt_layer=24,
                tgt_index=200,
                weight=3.14,
                count=5,
                weight_sum=15.7,
                weight_abs_sum=15.7,
                weight_sq_sum=49.3,
                weight_min=2.0,
                weight_max=4.0,
            ),
        ]
        count = db.write_edges(edges)
        assert count == 1

        # Read downstream edges from unit (10, 100)
        downstream = db.get_edges_for_unit(10, 100, direction="downstream")
        assert len(downstream) == 1
        assert downstream[0].tgt_layer == 24
        assert downstream[0].tgt_index == 200
        assert downstream[0].count == 5

        # Read upstream edges to unit (24, 200)
        upstream = db.get_edges_for_unit(24, 200, direction="upstream")
        assert len(upstream) == 1
        assert upstream[0].src_layer == 10

        # Read both directions for unit (10, 100)
        both = db.get_edges_for_unit(10, 100, direction="both")
        assert len(both) == 1  # only downstream edges exist for this unit

        db.close()

    def test_search_units_ilike(self, tmp_path):
        db = self._make_db(tmp_path)
        units = [
            Unit(layer=10, index=1, label="geography detector"),
            Unit(layer=15, index=2, label="enzyme pathway"),
            Unit(layer=20, index=3, label="European geography"),
        ]
        db.write_units(units)

        results = db.search_units("%geography%")
        assert len(results) == 2
        labels = {u.label for u in results}
        assert "geography detector" in labels
        assert "European geography" in labels

        results_enzyme = db.search_units("%enzyme%")
        assert len(results_enzyme) == 1
        assert results_enzyme[0].label == "enzyme pathway"

        db.close()

    def test_write_metadata_and_verify(self, tmp_path):
        db = self._make_db(tmp_path)
        db.write_metadata("model_name", "qwen3-32b")
        db.write_metadata("total_graphs", "12345")

        # Read back via raw query
        rows = db._conn.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
        meta = {k: v for k, v in rows}
        assert meta["model_name"] == "qwen3-32b"
        assert meta["total_graphs"] == "12345"

        # Test upsert
        db.write_metadata("model_name", "llama-3.1-8b")
        row = db._conn.execute(
            "SELECT value FROM metadata WHERE key = 'model_name'"
        ).fetchone()
        assert row[0] == "llama-3.1-8b"

        db.close()

    def test_get_units_by_cluster(self, tmp_path):
        db = self._make_db(tmp_path)
        units = [
            Unit(layer=10, index=1, label="a", top_cluster=42),
            Unit(layer=15, index=2, label="b", top_cluster=42),
            Unit(layer=20, index=3, label="c", top_cluster=99),
        ]
        db.write_units(units)

        cluster_42 = db.get_units_by_cluster(42)
        assert len(cluster_42) == 2
        layers = {u.layer for u in cluster_42}
        assert layers == {10, 15}

        cluster_99 = db.get_units_by_cluster(99)
        assert len(cluster_99) == 1

        cluster_none = db.get_units_by_cluster(0)
        assert len(cluster_none) == 0

        db.close()

    def test_write_read_roundtrip_full(self, tmp_path):
        db = self._make_db(tmp_path)
        units = [
            Unit(
                layer=24,
                index=5326,
                label="capital cities",
                input_label="France tokens",
                output_label="promotes Paris",
                max_activation=12.5,
                appearance_count=42,
                output_norm=0.75,
                cluster_path="1.5.3",
                top_cluster=1,
                sub_module=5,
                subsub_module=3,
                infomap_flow=0.002,
            ),
        ]
        db.write_units(units)

        u = db.get_unit(24, 5326)
        assert u is not None
        assert u.label == "capital cities"
        assert u.input_label == "France tokens"
        assert u.output_label == "promotes Paris"
        assert u.max_activation == pytest.approx(12.5)
        assert u.appearance_count == 42
        assert u.output_norm == pytest.approx(0.75)
        assert u.cluster_path == "1.5.3"
        assert u.top_cluster == 1
        assert u.sub_module == 5
        assert u.subsub_module == 3
        assert u.infomap_flow == pytest.approx(0.002)

        db.close()

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx_test.duckdb"
        with CircuitDatabase(db_path) as db:
            db.create_tables()
            db.write_units([Unit(layer=1, index=1, label="test")])
            u = db.get_unit(1, 1)
            assert u is not None
        # After context manager, connection should be closed
        assert db._conn is None


# =========================================================================
# 4. circuits.cli -- Argument parsing
# =========================================================================


class TestCLI:
    """CLI argument parser construction and basic parsing."""

    def test_build_parser_no_error(self):
        from circuits.cli import build_parser

        parser = build_parser()
        assert parser is not None
        assert parser.prog == "circuits"

    def test_parse_graph_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["graph", "What is the capital of France?"])
        assert args.command == "graph"
        assert args.prompt == "What is the capital of France?"
        assert args.k == 5
        assert args.tau == 0.005

    def test_parse_aggregate_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["aggregate", "/tmp/graphs", "--max-graphs", "100", "--fresh"]
        )
        assert args.command == "aggregate"
        assert args.graph_dir == "/tmp/graphs"
        assert args.max_graphs == 100
        assert args.fresh is True

    def test_parse_query_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["query", "db.duckdb", "enzyme", "--limit", "50"])
        assert args.command == "query"
        assert args.db == "db.duckdb"
        assert args.pattern == "enzyme"
        assert args.limit == 50
        assert args.sql is False

    def test_parse_query_sql_flag(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["query", "db.duckdb", "SELECT * FROM neurons LIMIT 5", "--sql"]
        )
        assert args.sql is True

    def test_parse_build_db_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "build-db",
                "/tmp/graphs",
                "--labels",
                "labels.jsonl",
                "--clusters",
                "clusters.json",
                "--min-edge-count",
                "5",
                "-o",
                "output.duckdb",
            ]
        )
        assert args.command == "build-db"
        assert args.graph_dir == "/tmp/graphs"
        assert args.labels == "labels.jsonl"
        assert args.clusters == "clusters.json"
        assert args.min_edge_count == 5
        assert args.output == "output.duckdb"

    def test_parse_analyze_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["analyze", "--config", "config.yaml", "-q"])
        assert args.command == "analyze"
        assert args.config == "config.yaml"
        assert args.quiet is True

    def test_parse_label_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "label",
                "--edge-stats",
                "stats.json",
                "--start-layer",
                "63",
                "--end-layer",
                "0",
                "--passes",
                "3",
            ]
        )
        assert args.command == "label"
        assert args.edge_stats == "stats.json"
        assert args.start_layer == 63
        assert args.end_layer == 0
        assert args.passes == 3

    def test_parse_cluster_subcommand(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "cluster",
                "--duckdb",
                "atlas.duckdb",
                "--min-edge-count",
                "10",
                "--weight-transform",
                "abs_weight_sq",
                "-o",
                "clusters.json",
            ]
        )
        assert args.command == "cluster"
        assert args.duckdb == "atlas.duckdb"
        assert args.min_edge_count == 10
        assert args.weight_transform == "abs_weight_sq"
        assert args.output == "clusters.json"

    def test_no_subcommand_returns_none(self):
        from circuits.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# =========================================================================
# Additional edge case tests
# =========================================================================


class TestTokenProjection:
    """TokenProjection roundtrip."""

    def test_roundtrip(self):
        tp = TokenProjection(token="Paris", token_id=3681, weight=2.5)
        d = tp.to_dict()
        tp2 = TokenProjection.from_dict(d)
        assert tp2.token == "Paris"
        assert tp2.token_id == 3681
        assert tp2.weight == pytest.approx(2.5)


class TestUnitLabel:
    """UnitLabel roundtrip."""

    def test_roundtrip(self):
        lbl = UnitLabel(
            text="geography",
            source=LabelSource.NEURONDB,
            confidence="high",
            description="Detects geographical references",
        )
        d = lbl.to_dict()
        lbl2 = UnitLabel.from_dict(d)
        assert lbl2.text == "geography"
        assert lbl2.source == LabelSource.NEURONDB
        assert lbl2.confidence == "high"
        assert lbl2.description == "Detects geographical references"

    def test_defaults(self):
        lbl = UnitLabel(text="test", source=LabelSource.HUMAN)
        assert lbl.confidence == "unknown"
        assert lbl.description == ""


class TestEnums:
    """Verify enum values."""

    def test_unit_type_values(self):
        assert UnitType.NEURON.value == "neuron"
        assert UnitType.TRANSCODER.value == "transcoder"
        assert UnitType.ATTENTION_HEAD.value == "attention_head"

    def test_label_source_values(self):
        assert LabelSource.NEURONDB.value == "neurondb"
        assert LabelSource.AUTOINTERP.value == "autointerp"
        assert LabelSource.HUMAN.value == "human"

    def test_connectivity_method_values(self):
        assert ConnectivityMethod.RELP.value == "relp"
        assert ConnectivityMethod.OUTPUT_PROJECTION.value == "output_projection"
        assert ConnectivityMethod.WEIGHT_GRAPH.value == "weight_graph"
