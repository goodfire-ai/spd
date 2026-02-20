#!/usr/bin/env python3
"""
Graph Diff Tool - Compare two attribution graphs and compute similarity metrics.

Usage:
    python scripts/graph_diff.py graph_A.json graph_B.json [--output diff.json] [--clusters-a clusters_A.json] [--clusters-b clusters_B.json]

Metrics computed:
- Jaccard similarity (neuron sets)
- Weighted Jaccard similarity (influence-weighted)
- Edge Jaccard similarity
- Flow conservation score
- Spectral similarity
- Module alignment score

Diffs produced:
- Node diff (unique to each, differential activation)
- Edge diff (unique edges, weight changes)
- Structural diff (module changes, path analysis)
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment


@dataclass
class NeuronInfo:
    """Information about a neuron node."""
    layer: int
    neuron: int
    position: int
    influence: float
    activation: float
    token: str = ""
    node_id: str = ""

    @property
    def key(self) -> str:
        return f"L{self.layer}/N{self.neuron}"


@dataclass
class EdgeInfo:
    """Information about an edge."""
    source: str
    target: str
    weight: float

    @property
    def key(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass
class ClusterInfo:
    """Information about a cluster/module."""
    cluster_id: int
    members: list
    total_influence: float = 0.0
    layer_range: tuple = (0, 0)

    def get_neuron_keys(self) -> set:
        return {f"L{m['layer']}/N{m['neuron']}" for m in self.members}


@dataclass
class SimilarityMetrics:
    """Container for all similarity metrics."""
    jaccard_neurons: float = 0.0
    weighted_jaccard_neurons: float = 0.0
    jaccard_edges: float = 0.0
    flow_conservation: float = 0.0
    spectral_similarity: float = 0.0
    module_alignment: float = 0.0


@dataclass
class NodeDiff:
    """Diff information for nodes."""
    only_in_a: list = field(default_factory=list)
    only_in_b: list = field(default_factory=list)
    shared_with_diff: list = field(default_factory=list)  # Significant influence changes
    layer_divergence: dict = field(default_factory=dict)  # Per-layer stats


@dataclass
class EdgeDiff:
    """Diff information for edges."""
    only_in_a: list = field(default_factory=list)
    only_in_b: list = field(default_factory=list)
    weight_changes: list = field(default_factory=list)  # Significant weight changes


@dataclass
class StructuralDiff:
    """Structural differences."""
    module_changes: dict = field(default_factory=dict)
    path_changes: dict = field(default_factory=dict)


@dataclass
class GraphDiffResult:
    """Complete diff result."""
    graph_a: str
    graph_b: str
    similarity: SimilarityMetrics = field(default_factory=SimilarityMetrics)
    node_diff: NodeDiff = field(default_factory=NodeDiff)
    edge_diff: EdgeDiff = field(default_factory=EdgeDiff)
    structural_diff: StructuralDiff = field(default_factory=StructuralDiff)
    metadata: dict = field(default_factory=dict)


def load_graph(path: str) -> dict:
    """Load a graph JSON file."""
    with open(path) as f:
        return json.load(f)


def load_clusters(path: str) -> list:
    """Load clusters from a clusters JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Handle the nested structure
    if 'methods' in data and isinstance(data['methods'], list):
        # Find infomap or first method
        for method in data['methods']:
            if method.get('method') == 'infomap':
                return method.get('clusters', [])
        # Fallback to first method
        if data['methods']:
            return data['methods'][0].get('clusters', [])
    return []


def extract_neurons(graph: dict) -> dict[str, NeuronInfo]:
    """Extract all neuron nodes from a graph."""
    neurons = {}
    for node in graph.get('nodes', []):
        layer = node.get('layer')
        # Skip non-neuron nodes (embeddings, logits)
        if layer == 'E' or not layer:
            continue
        try:
            layer_int = int(layer)
        except (ValueError, TypeError):
            continue

        # Skip logit nodes (very high layer numbers)
        if layer_int > 31:
            continue

        neuron = NeuronInfo(
            layer=layer_int,
            neuron=node.get('feature', 0),
            position=node.get('ctx_idx', -1),
            influence=node.get('influence', 0) or 0,
            activation=node.get('activation', 0) or 0,
            token=node.get('clerp', '').replace('Token: ', ''),
            node_id=node.get('node_id', '')
        )
        neurons[neuron.key] = neuron
    return neurons


def extract_edges(graph: dict) -> dict[str, EdgeInfo]:
    """Extract all edges from a graph."""
    edges = {}
    for link in graph.get('links', []):
        edge = EdgeInfo(
            source=link.get('source', ''),
            target=link.get('target', ''),
            weight=link.get('weight', 0)
        )
        edges[edge.key] = edge
    return edges


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_weighted_jaccard(neurons_a: dict[str, NeuronInfo],
                             neurons_b: dict[str, NeuronInfo]) -> float:
    """Compute influence-weighted Jaccard similarity."""
    keys_a = set(neurons_a.keys())
    keys_b = set(neurons_b.keys())

    if not keys_a and not keys_b:
        return 1.0

    # For intersection: use minimum influence
    # For union: use maximum influence
    intersection_weight = 0.0
    union_weight = 0.0

    all_keys = keys_a | keys_b
    for key in all_keys:
        inf_a = abs(neurons_a[key].influence) if key in neurons_a else 0
        inf_b = abs(neurons_b[key].influence) if key in neurons_b else 0

        intersection_weight += min(inf_a, inf_b)
        union_weight += max(inf_a, inf_b)

    return intersection_weight / union_weight if union_weight > 0 else 0.0


def compute_edge_jaccard(edges_a: dict[str, EdgeInfo],
                         edges_b: dict[str, EdgeInfo]) -> float:
    """Compute Jaccard similarity for edges."""
    keys_a = set(edges_a.keys())
    keys_b = set(edges_b.keys())
    return compute_jaccard(keys_a, keys_b)


def compute_flow_conservation(graph_a: dict, graph_b: dict,
                              neurons_a: dict, neurons_b: dict) -> float:
    """
    Compute flow conservation score.

    Measures how well the flow patterns are preserved between graphs.
    For shared neurons, we compare their incoming and outgoing flow.
    """
    edges_a = graph_a.get('links', [])
    edges_b = graph_b.get('links', [])

    # Build flow maps: node -> (total_in, total_out)
    def build_flow_map(edges, neurons):
        flow = defaultdict(lambda: [0.0, 0.0])  # [in, out]
        for e in edges:
            src = e.get('source', '')
            tgt = e.get('target', '')
            w = abs(e.get('weight', 0))

            # Map node_ids to neuron keys if possible
            # node_id format: "layer_neuron_position"
            src_parts = src.split('_')
            tgt_parts = tgt.split('_')

            if len(src_parts) >= 2:
                src_key = f"L{src_parts[0]}/N{src_parts[1]}"
                if src_key in neurons:
                    flow[src_key][1] += w  # outgoing

            if len(tgt_parts) >= 2:
                tgt_key = f"L{tgt_parts[0]}/N{tgt_parts[1]}"
                if tgt_key in neurons:
                    flow[tgt_key][0] += w  # incoming
        return flow

    flow_a = build_flow_map(edges_a, neurons_a)
    flow_b = build_flow_map(edges_b, neurons_b)

    # Compare flow patterns for shared neurons
    shared = set(neurons_a.keys()) & set(neurons_b.keys())
    if not shared:
        return 0.0

    correlations = []
    for key in shared:
        in_a, out_a = flow_a.get(key, [0, 0])
        in_b, out_b = flow_b.get(key, [0, 0])

        # Compute cosine similarity of flow vectors
        vec_a = np.array([in_a, out_a])
        vec_b = np.array([in_b, out_b])

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a > 0 and norm_b > 0:
            cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            correlations.append(cos_sim)

    return np.mean(correlations) if correlations else 0.0


def build_adjacency_matrix(graph: dict, neurons: dict) -> tuple[np.ndarray, list]:
    """Build adjacency matrix for spectral analysis."""
    # Create index mapping
    neuron_keys = sorted(neurons.keys())
    key_to_idx = {k: i for i, k in enumerate(neuron_keys)}
    n = len(neuron_keys)

    if n == 0:
        return np.array([[]]), []

    # Build sparse adjacency matrix
    rows, cols, data = [], [], []

    for link in graph.get('links', []):
        src = link.get('source', '')
        tgt = link.get('target', '')
        w = link.get('weight', 0)

        # Parse node_ids
        src_parts = src.split('_')
        tgt_parts = tgt.split('_')

        if len(src_parts) >= 2 and len(tgt_parts) >= 2:
            src_key = f"L{src_parts[0]}/N{src_parts[1]}"
            tgt_key = f"L{tgt_parts[0]}/N{tgt_parts[1]}"

            if src_key in key_to_idx and tgt_key in key_to_idx:
                rows.append(key_to_idx[src_key])
                cols.append(key_to_idx[tgt_key])
                data.append(abs(w))

    if not rows:
        return np.zeros((n, n)), neuron_keys

    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Make symmetric for spectral analysis
    adj = adj + adj.T

    return adj.toarray(), neuron_keys


def compute_spectral_similarity(graph_a: dict, graph_b: dict,
                                neurons_a: dict, neurons_b: dict,
                                k: int = 10) -> float:
    """
    Compute spectral similarity using eigenvalues of the Laplacian.

    Uses the k smallest non-zero eigenvalues of the normalized Laplacian.
    """
    def get_laplacian_eigenvalues(adj: np.ndarray, k: int) -> np.ndarray:
        n = adj.shape[0]
        if n < 2:
            return np.array([])

        # Degree matrix
        degrees = np.sum(adj, axis=1)
        degrees = np.where(degrees > 0, degrees, 1)  # Avoid division by zero

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        laplacian = np.eye(n) - d_inv_sqrt @ adj @ d_inv_sqrt

        # Get eigenvalues
        k_actual = min(k, n - 1)
        if k_actual < 1:
            return np.array([])

        try:
            eigenvalues = np.linalg.eigvalsh(laplacian)
            # Sort and take smallest k (excluding the zero eigenvalue)
            eigenvalues = np.sort(eigenvalues)
            return eigenvalues[1:k_actual+1]  # Skip the first (zero) eigenvalue
        except:
            return np.array([])

    adj_a, _ = build_adjacency_matrix(graph_a, neurons_a)
    adj_b, _ = build_adjacency_matrix(graph_b, neurons_b)

    eig_a = get_laplacian_eigenvalues(adj_a, k)
    eig_b = get_laplacian_eigenvalues(adj_b, k)

    if len(eig_a) == 0 or len(eig_b) == 0:
        return 0.0

    # Pad to same length
    max_len = max(len(eig_a), len(eig_b))
    eig_a = np.pad(eig_a, (0, max_len - len(eig_a)))
    eig_b = np.pad(eig_b, (0, max_len - len(eig_b)))

    # Compute similarity as 1 / (1 + distance)
    distance = np.linalg.norm(eig_a - eig_b)
    return 1.0 / (1.0 + distance)


def compute_module_alignment(clusters_a: list, clusters_b: list) -> float:
    """
    Compute module alignment score using Hungarian algorithm.

    Finds optimal matching between modules and computes overlap score.
    """
    if not clusters_a or not clusters_b:
        return 0.0

    # Get neuron sets for each cluster
    sets_a = [set(f"L{m['layer']}/N{m['neuron']}" for m in c.get('members', []))
              for c in clusters_a]
    sets_b = [set(f"L{m['layer']}/N{m['neuron']}" for m in c.get('members', []))
              for c in clusters_b]

    # Filter out empty clusters
    sets_a = [s for s in sets_a if s]
    sets_b = [s for s in sets_b if s]

    if not sets_a or not sets_b:
        return 0.0

    n_a, n_b = len(sets_a), len(sets_b)

    # Build cost matrix (negative Jaccard for maximization via Hungarian)
    cost_matrix = np.zeros((n_a, n_b))
    for i, sa in enumerate(sets_a):
        for j, sb in enumerate(sets_b):
            jaccard = compute_jaccard(sa, sb)
            cost_matrix[i, j] = -jaccard  # Negative for minimization

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute average Jaccard of matched pairs
    total_jaccard = 0.0
    for i, j in zip(row_ind, col_ind):
        total_jaccard += -cost_matrix[i, j]

    # Normalize by max number of clusters
    return total_jaccard / max(n_a, n_b)


def compute_node_diff(neurons_a: dict[str, NeuronInfo],
                      neurons_b: dict[str, NeuronInfo],
                      influence_threshold: float = 0.5) -> NodeDiff:
    """Compute detailed node differences."""
    keys_a = set(neurons_a.keys())
    keys_b = set(neurons_b.keys())

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    shared = keys_a & keys_b

    diff = NodeDiff()

    # Nodes only in A
    diff.only_in_a = sorted([
        {
            'key': k,
            'layer': neurons_a[k].layer,
            'neuron': neurons_a[k].neuron,
            'influence': neurons_a[k].influence,
            'position': neurons_a[k].position,
            'token': neurons_a[k].token
        }
        for k in only_a
    ], key=lambda x: abs(x['influence']), reverse=True)

    # Nodes only in B
    diff.only_in_b = sorted([
        {
            'key': k,
            'layer': neurons_b[k].layer,
            'neuron': neurons_b[k].neuron,
            'influence': neurons_b[k].influence,
            'position': neurons_b[k].position,
            'token': neurons_b[k].token
        }
        for k in only_b
    ], key=lambda x: abs(x['influence']), reverse=True)

    # Shared nodes with significant influence changes
    for k in shared:
        inf_a = neurons_a[k].influence
        inf_b = neurons_b[k].influence
        delta = inf_a - inf_b

        if abs(delta) >= influence_threshold:
            # Get description, prefer longer one (more likely to be meaningful)
            token_a = neurons_a[k].token
            token_b = neurons_b[k].token
            token = token_a if len(token_a) >= len(token_b) else token_b
            diff.shared_with_diff.append({
                'key': k,
                'layer': neurons_a[k].layer,
                'neuron': neurons_a[k].neuron,
                'influence_a': inf_a,
                'influence_b': inf_b,
                'delta': delta,
                'position': neurons_a[k].position,
                'token': token
            })

    diff.shared_with_diff.sort(key=lambda x: abs(x['delta']), reverse=True)

    # Layer-by-layer divergence
    layer_stats = defaultdict(lambda: {'only_a': 0, 'only_b': 0, 'shared': 0, 'total_delta': 0.0})

    for k in only_a:
        layer_stats[neurons_a[k].layer]['only_a'] += 1
    for k in only_b:
        layer_stats[neurons_b[k].layer]['only_b'] += 1
    for k in shared:
        layer = neurons_a[k].layer
        layer_stats[layer]['shared'] += 1
        layer_stats[layer]['total_delta'] += abs(neurons_a[k].influence - neurons_b[k].influence)

    diff.layer_divergence = dict(sorted(layer_stats.items()))

    return diff


def compute_edge_diff(edges_a: dict[str, EdgeInfo],
                      edges_b: dict[str, EdgeInfo],
                      weight_threshold: float = 1.0) -> EdgeDiff:
    """Compute detailed edge differences."""
    keys_a = set(edges_a.keys())
    keys_b = set(edges_b.keys())

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    shared = keys_a & keys_b

    diff = EdgeDiff()

    # Edges only in A (top by weight)
    diff.only_in_a = sorted([
        {
            'key': k,
            'source': edges_a[k].source,
            'target': edges_a[k].target,
            'weight': edges_a[k].weight
        }
        for k in only_a
    ], key=lambda x: abs(x['weight']), reverse=True)[:50]  # Top 50

    # Edges only in B
    diff.only_in_b = sorted([
        {
            'key': k,
            'source': edges_b[k].source,
            'target': edges_b[k].target,
            'weight': edges_b[k].weight
        }
        for k in only_b
    ], key=lambda x: abs(x['weight']), reverse=True)[:50]

    # Significant weight changes
    for k in shared:
        w_a = edges_a[k].weight
        w_b = edges_b[k].weight
        delta = w_a - w_b

        if abs(delta) >= weight_threshold:
            diff.weight_changes.append({
                'key': k,
                'source': edges_a[k].source,
                'target': edges_a[k].target,
                'weight_a': w_a,
                'weight_b': w_b,
                'delta': delta
            })

    diff.weight_changes.sort(key=lambda x: abs(x['delta']), reverse=True)
    diff.weight_changes = diff.weight_changes[:50]  # Top 50

    return diff


def compute_structural_diff(clusters_a: list, clusters_b: list,
                           neurons_a: dict, neurons_b: dict,
                           graph_a: dict, graph_b: dict) -> StructuralDiff:
    """Compute structural differences between graphs."""
    diff = StructuralDiff()

    # Module changes
    if clusters_a and clusters_b:
        # Get neuron sets
        sets_a = {i: set(f"L{m['layer']}/N{m['neuron']}" for m in c.get('members', []))
                  for i, c in enumerate(clusters_a)}
        sets_b = {i: set(f"L{m['layer']}/N{m['neuron']}" for m in c.get('members', []))
                  for i, c in enumerate(clusters_b)}

        # Find best matches using Jaccard
        module_mapping = {}
        for i, sa in sets_a.items():
            if not sa:
                continue
            best_j, best_score = -1, 0
            for j, sb in sets_b.items():
                if not sb:
                    continue
                score = compute_jaccard(sa, sb)
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j >= 0:
                module_mapping[i] = {
                    'matched_to': best_j,
                    'jaccard': best_score,
                    'neurons_lost': list(sa - sets_b.get(best_j, set())),
                    'neurons_gained': list(sets_b.get(best_j, set()) - sa),
                    'size_a': len(sa),
                    'size_b': len(sets_b.get(best_j, set()))
                }

        diff.module_changes = {
            'n_modules_a': len(clusters_a),
            'n_modules_b': len(clusters_b),
            'mappings': module_mapping
        }

    # Path analysis: find key pathways and see if they're preserved
    # We look at high-influence paths through the network
    def get_key_paths(graph: dict, neurons: dict, top_k: int = 10) -> list:
        """Get top-k paths by total weight."""
        # Build adjacency with weights
        edges = defaultdict(list)
        for link in graph.get('links', []):
            src = link.get('source', '')
            tgt = link.get('target', '')
            w = link.get('weight', 0)

            src_parts = src.split('_')
            tgt_parts = tgt.split('_')

            if len(src_parts) >= 2 and len(tgt_parts) >= 2:
                src_key = f"L{src_parts[0]}/N{src_parts[1]}"
                tgt_key = f"L{tgt_parts[0]}/N{tgt_parts[1]}"

                if src_key in neurons and tgt_key in neurons:
                    edges[src_key].append((tgt_key, w))

        # Find high-weight 2-hop paths
        paths = []
        for src, targets in edges.items():
            for mid, w1 in targets:
                if mid in edges:
                    for dst, w2 in edges[mid]:
                        paths.append((src, mid, dst, w1 + w2))

        paths.sort(key=lambda x: abs(x[3]), reverse=True)
        return paths[:top_k]

    paths_a = get_key_paths(graph_a, neurons_a)
    paths_b = get_key_paths(graph_b, neurons_b)

    # Convert to comparable format
    path_set_a = set((p[0], p[1], p[2]) for p in paths_a)
    path_set_b = set((p[0], p[1], p[2]) for p in paths_b)

    diff.path_changes = {
        'paths_only_in_a': [{'path': list(p)} for p in (path_set_a - path_set_b)],
        'paths_only_in_b': [{'path': list(p)} for p in (path_set_b - path_set_a)],
        'paths_preserved': [{'path': list(p)} for p in (path_set_a & path_set_b)],
        'preservation_rate': len(path_set_a & path_set_b) / len(path_set_a | path_set_b)
                           if path_set_a | path_set_b else 1.0
    }

    return diff


def compute_decision_circuit(graph_a: dict, graph_b: dict) -> dict:
    """
    Analyze edges to logit nodes to understand what drives the decision.

    Returns dict with:
    - logit_nodes: mapping of logit node IDs to tokens
    - edge_diffs: edges with largest weight differences to logits
    """
    # Find logit nodes (node_id starts with 'L_')
    def get_logit_map(graph):
        logit_map = {}
        for node in graph['nodes']:
            node_id = node.get('node_id', '')
            if node_id.startswith('L_'):
                clerp = node.get('clerp', '')
                # Extract token from clerp like ' yes (p=0.4531)'
                token = clerp.split(' (p=')[0].strip()
                prob = 0.0
                if '(p=' in clerp:
                    try:
                        prob = float(clerp.split('(p=')[1].rstrip(')'))
                    except:
                        pass
                logit_map[node_id] = {'token': token, 'probability': prob}
        return logit_map

    logits_a = get_logit_map(graph_a)
    logits_b = get_logit_map(graph_b)

    # Build edge dicts
    def build_edge_dict(graph):
        edges = {}
        for link in graph['links']:
            key = f"{link['source']}->{link['target']}"
            edges[key] = link.get('weight', 0)
        return edges

    edges_a = build_edge_dict(graph_a)
    edges_b = build_edge_dict(graph_b)

    # Get neuron info
    def get_neuron_clerp(graph, node_id):
        for node in graph['nodes']:
            if node.get('node_id') == node_id:
                clerp = node.get('clerp', '')
                if ': ' in clerp:
                    return clerp.split(': ', 1)[1]
                return clerp
        return ''

    # Find all logit node IDs
    all_logit_ids = set(logits_a.keys()) | set(logits_b.keys())

    # Compute edge weight differences to logits
    edge_diffs = []
    all_edges = set(edges_a.keys()) | set(edges_b.keys())

    for e in all_edges:
        src, tgt = e.split('->')
        if tgt in all_logit_ids:
            w_a = edges_a.get(e, 0)
            w_b = edges_b.get(e, 0)
            if abs(w_a - w_b) > 0.5:  # Significant difference
                token_info = logits_a.get(tgt) or logits_b.get(tgt, {})
                clerp = get_neuron_clerp(graph_a, src) or get_neuron_clerp(graph_b, src)
                edge_diffs.append({
                    'source': src,
                    'target': tgt,
                    'token': token_info.get('token', ''),
                    'weight_a': w_a,
                    'weight_b': w_b,
                    'diff': w_a - w_b,
                    'source_description': clerp[:80] if clerp else ''
                })

    # Sort by absolute difference
    edge_diffs.sort(key=lambda x: abs(x['diff']), reverse=True)

    return {
        'logits_a': {k: v for k, v in logits_a.items()},
        'logits_b': {k: v for k, v in logits_b.items()},
        'edge_diffs_to_logits': edge_diffs[:30]  # Top 30
    }


def diff_graphs(graph_a_path: str, graph_b_path: str,
                clusters_a_path: str | None = None,
                clusters_b_path: str | None = None) -> GraphDiffResult:
    """Main function to compute full graph diff."""

    # Load graphs
    graph_a = load_graph(graph_a_path)
    graph_b = load_graph(graph_b_path)

    # Load clusters if provided
    clusters_a = load_clusters(clusters_a_path) if clusters_a_path else []
    clusters_b = load_clusters(clusters_b_path) if clusters_b_path else []

    # Extract neurons and edges
    neurons_a = extract_neurons(graph_a)
    neurons_b = extract_neurons(graph_b)
    edges_a = extract_edges(graph_a)
    edges_b = extract_edges(graph_b)

    # Initialize result
    result = GraphDiffResult(
        graph_a=graph_a_path,
        graph_b=graph_b_path
    )

    # Compute similarity metrics
    result.similarity.jaccard_neurons = compute_jaccard(
        set(neurons_a.keys()), set(neurons_b.keys())
    )
    result.similarity.weighted_jaccard_neurons = compute_weighted_jaccard(
        neurons_a, neurons_b
    )
    result.similarity.jaccard_edges = compute_edge_jaccard(edges_a, edges_b)
    result.similarity.flow_conservation = compute_flow_conservation(
        graph_a, graph_b, neurons_a, neurons_b
    )
    result.similarity.spectral_similarity = compute_spectral_similarity(
        graph_a, graph_b, neurons_a, neurons_b
    )
    result.similarity.module_alignment = compute_module_alignment(
        clusters_a, clusters_b
    )

    # Compute diffs
    result.node_diff = compute_node_diff(neurons_a, neurons_b)
    result.edge_diff = compute_edge_diff(edges_a, edges_b)
    result.structural_diff = compute_structural_diff(
        clusters_a, clusters_b, neurons_a, neurons_b, graph_a, graph_b
    )

    # Compute decision circuit analysis
    decision_circuit = compute_decision_circuit(graph_a, graph_b)

    # Add metadata
    result.metadata = {
        'n_neurons_a': len(neurons_a),
        'n_neurons_b': len(neurons_b),
        'n_edges_a': len(edges_a),
        'n_edges_b': len(edges_b),
        'n_clusters_a': len(clusters_a),
        'n_clusters_b': len(clusters_b),
        'prompt_a': graph_a.get('metadata', {}).get('prompt', ''),
        'prompt_b': graph_b.get('metadata', {}).get('prompt', ''),
        'decision_circuit': decision_circuit
    }

    return result


def result_to_dict(result: GraphDiffResult) -> dict:
    """Convert result to JSON-serializable dict."""
    return {
        'graph_a': result.graph_a,
        'graph_b': result.graph_b,
        'similarity': asdict(result.similarity),
        'node_diff': {
            'only_in_a': result.node_diff.only_in_a,
            'only_in_b': result.node_diff.only_in_b,
            'shared_with_diff': result.node_diff.shared_with_diff,
            'layer_divergence': result.node_diff.layer_divergence
        },
        'edge_diff': {
            'only_in_a': result.edge_diff.only_in_a,
            'only_in_b': result.edge_diff.only_in_b,
            'weight_changes': result.edge_diff.weight_changes
        },
        'structural_diff': {
            'module_changes': result.structural_diff.module_changes,
            'path_changes': result.structural_diff.path_changes
        },
        'metadata': result.metadata
    }


def print_summary(result: GraphDiffResult):
    """Print a human-readable summary."""
    print("=" * 70)
    print("GRAPH DIFF SUMMARY")
    print("=" * 70)

    print(f"\nGraph A: {result.graph_a}")
    print(f"Graph B: {result.graph_b}")

    print("\n--- Similarity Metrics ---")
    print(f"  Jaccard (neurons):          {result.similarity.jaccard_neurons:.3f}")
    print(f"  Weighted Jaccard (neurons): {result.similarity.weighted_jaccard_neurons:.3f}")
    print(f"  Jaccard (edges):            {result.similarity.jaccard_edges:.3f}")
    print(f"  Flow conservation:          {result.similarity.flow_conservation:.3f}")
    print(f"  Spectral similarity:        {result.similarity.spectral_similarity:.3f}")
    print(f"  Module alignment:           {result.similarity.module_alignment:.3f}")

    print("\n--- Node Statistics ---")
    print(f"  Neurons in A: {result.metadata['n_neurons_a']}")
    print(f"  Neurons in B: {result.metadata['n_neurons_b']}")
    print(f"  Only in A: {len(result.node_diff.only_in_a)}")
    print(f"  Only in B: {len(result.node_diff.only_in_b)}")
    print(f"  Shared with significant diff: {len(result.node_diff.shared_with_diff)}")

    def format_description(token: str, max_len: int = 80) -> str:
        """Format neuron description, removing key prefix if present."""
        # Remove "L##/N#####: " prefix if present
        if ': ' in token and token.startswith('L'):
            token = token.split(': ', 1)[1]
        if len(token) > max_len:
            return token[:max_len-3] + '...'
        return token

    print("\n--- Top Nodes Only in A ---")
    for n in result.node_diff.only_in_a[:5]:
        desc = format_description(n['token'])
        print(f"  {n['key']}: inf={n['influence']:.2f}")
        if desc:
            print(f"    → {desc}")

    print("\n--- Top Nodes Only in B ---")
    for n in result.node_diff.only_in_b[:5]:
        desc = format_description(n['token'])
        print(f"  {n['key']}: inf={n['influence']:.2f}")
        if desc:
            print(f"    → {desc}")

    print("\n--- Top Differential Activations ---")
    for n in result.node_diff.shared_with_diff[:5]:
        direction = "→B" if n['delta'] < 0 else "→A"
        desc = format_description(n['token'])
        print(f"  {n['key']}: Δ={n['delta']:+.2f} (A={n['influence_a']:.2f}, B={n['influence_b']:.2f}) {direction}")
        if desc:
            print(f"    → {desc}")

    print("\n--- Layer Divergence ---")
    for layer, stats in sorted(result.node_diff.layer_divergence.items()):
        if stats['only_a'] > 0 or stats['only_b'] > 0:
            print(f"  L{layer}: only_A={stats['only_a']}, only_B={stats['only_b']}, "
                  f"shared={stats['shared']}, total_Δ={stats['total_delta']:.2f}")

    if result.structural_diff.path_changes:
        pc = result.structural_diff.path_changes
        print("\n--- Path Preservation ---")
        print(f"  Paths preserved: {len(pc.get('paths_preserved', []))}")
        print(f"  Paths only in A: {len(pc.get('paths_only_in_a', []))}")
        print(f"  Paths only in B: {len(pc.get('paths_only_in_b', []))}")
        print(f"  Preservation rate: {pc.get('preservation_rate', 0):.3f}")

    # Decision circuit analysis
    dc = result.metadata.get('decision_circuit', {})
    if dc:
        print("\n--- Decision Circuit (edges to output logits) ---")
        logits_a = dc.get('logits_a', {})
        logits_b = dc.get('logits_b', {})
        logits_a_str = ', '.join(f"{v['token']} ({v['probability']*100:.1f}%)" for v in logits_a.values())
        logits_b_str = ', '.join(f"{v['token']} ({v['probability']*100:.1f}%)" for v in logits_b.values())
        print(f"  Graph A logits: {logits_a_str}")
        print(f"  Graph B logits: {logits_b_str}")

        edge_diffs = dc.get('edge_diffs_to_logits', [])
        if edge_diffs:
            print("\n  Top edges with different weights to logits:")
            for e in edge_diffs[:10]:
                direction = "→A" if e['diff'] > 0 else "→B"
                print(f"    {e['source']} --> '{e['token']}': A={e['weight_a']:+.2f}, B={e['weight_b']:+.2f}, Δ={e['diff']:+.2f} {direction}")
                if e['source_description']:
                    print(f"      [{e['source_description'][:60]}]")


def main():
    parser = argparse.ArgumentParser(description='Compare two attribution graphs')
    parser.add_argument('graph_a', help='Path to first graph JSON')
    parser.add_argument('graph_b', help='Path to second graph JSON')
    parser.add_argument('--clusters-a', help='Path to clusters for graph A')
    parser.add_argument('--clusters-b', help='Path to clusters for graph B')
    parser.add_argument('--output', '-o', help='Output JSON path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress summary output')

    args = parser.parse_args()

    # Auto-detect cluster files if not provided
    if not args.clusters_a:
        clusters_a_path = args.graph_a.replace('-graph.json', '-clusters.json')
        if clusters_a_path != args.graph_a:
            try:
                with open(clusters_a_path):
                    args.clusters_a = clusters_a_path
            except FileNotFoundError:
                pass

    if not args.clusters_b:
        clusters_b_path = args.graph_b.replace('-graph.json', '-clusters.json')
        if clusters_b_path != args.graph_b:
            try:
                with open(clusters_b_path):
                    args.clusters_b = clusters_b_path
            except FileNotFoundError:
                pass

    # Run diff
    result = diff_graphs(
        args.graph_a, args.graph_b,
        args.clusters_a, args.clusters_b
    )

    # Output
    if not args.quiet:
        print_summary(result)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result_to_dict(result), f, indent=2)
        print(f"\nFull diff saved to: {args.output}")

    return result


if __name__ == '__main__':
    main()
