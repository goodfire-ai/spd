"""Infomap-based clustering for attribution graphs.

Clusters neurons using flow-based community detection.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .schemas import Edge, Graph, Unit

try:
    import infomap
    HAS_INFOMAP = True
except ImportError:
    HAS_INFOMAP = False


@dataclass
class HierarchicalCluster:
    """Represents a cluster in the hierarchy."""
    cluster_id: str
    nodes: list[dict[str, Any]]
    children: list['HierarchicalCluster'] = field(default_factory=list)
    depth: int = 0

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def total_size(self) -> int:
        return self.size + sum(c.total_size for c in self.children)

    def get_all_nodes(self) -> list[dict[str, Any]]:
        all_nodes = list(self.nodes)
        for child in self.children:
            all_nodes.extend(child.get_all_nodes())
        return all_nodes


def get_special_token_positions(graph_data: dict) -> set:
    """Identify positions of special tokens (BOS, headers, etc.)."""
    special_positions = set()
    metadata = graph_data.get('metadata', {})
    prompt_tokens = metadata.get('prompt_tokens', [])

    special_patterns = [
        '<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>',
        '<|eot_id|>', 'system', 'user', 'assistant',
    ]

    for i, token in enumerate(prompt_tokens):
        if any(pat in token for pat in special_patterns) or token in ['', '\n', '\\n', 'Ċ', '\u010a'] and i < 10:
            special_positions.add(i)

    return special_positions


def extract_neurons_and_logits(
    graph_data: dict,
    skip_special_positions: set | None = None
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Extract neurons, node lookup, and logit nodes from graph.

    Returns:
        Tuple of (neurons, node_lookup, logit_nodes)
    """
    skip_special_positions = skip_special_positions or set()

    neurons = []
    logit_nodes = []
    node_lookup = {}

    for node in graph_data['nodes']:
        node_lookup[node['node_id']] = node
        layer = node.get('layer', '')
        feature_type = node.get('feature_type', '')

        if layer == 'E' or feature_type == 'embedding':
            continue
        elif node.get('isLogit', False) or feature_type == 'logit':
            logit_nodes.append(node)
            continue
        elif str(layer).isdigit():
            pos = node.get('ctx_idx', node.get('position'))
            if pos is not None and pos in skip_special_positions:
                continue
            neurons.append(node)

    # Sort logit nodes by probability
    def get_prob(node):
        clerp = node.get('clerp', '')
        match = re.search(r'p=([0-9.]+)', clerp)
        return float(match.group(1)) if match else 0
    logit_nodes.sort(key=get_prob, reverse=True)

    return neurons, node_lookup, logit_nodes


def get_node_position(node: dict) -> int:
    """Get token position from node."""
    return node.get('ctx_idx', node.get('position', 0))


def run_infomap(
    nodes: list[dict],
    edges: list[dict],
    node_lookup: dict[str, dict],
    num_trials: int = 10
) -> dict[int, list[dict]]:
    """Run two-level Infomap clustering."""
    if not HAS_INFOMAP:
        raise ImportError("infomap not installed. Run: uv pip install infomap")

    if len(nodes) < 2:
        return {0: nodes}

    node_ids = [n['node_id'] for n in nodes]
    node_set = set(node_ids)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_to_node = {i: node_lookup[nid] for i, nid in enumerate(node_ids)}

    im = infomap.Infomap(f'--two-level --directed --num-trials {num_trials} --silent')

    link_count = 0
    for edge in edges:
        src, tgt = edge.get('source'), edge.get('target')
        if src in node_set and tgt in node_set:
            weight = abs(edge.get('weight', 1.0))
            if weight > 0:
                im.add_link(node_to_idx[src], node_to_idx[tgt], weight)
                link_count += 1

    # If no links were added, return all nodes in a single cluster
    if link_count == 0:
        return {0: nodes}

    im.run()

    clusters = defaultdict(list)
    for node_id in im.tree:
        if node_id.is_leaf:
            cluster_id = node_id.module_id
            original_node = idx_to_node[node_id.node_id]
            clusters[cluster_id].append(original_node)

    return dict(clusters)


def recursive_infomap(
    nodes: list[dict],
    edges: list[dict],
    node_lookup: dict[str, dict],
    min_cluster_size: int = 10,
    max_depth: int = 3,
    current_depth: int = 0,
    parent_id: str = ""
) -> list[HierarchicalCluster]:
    """Recursively apply Infomap to subdivide large clusters."""
    initial_clusters = run_infomap(nodes, edges, node_lookup)

    result = []
    for cluster_idx, cluster_nodes in initial_clusters.items():
        cluster_id = f"{parent_id}{cluster_idx + 1}" if parent_id else str(cluster_idx + 1)

        cluster = HierarchicalCluster(
            cluster_id=cluster_id,
            nodes=cluster_nodes,
            depth=current_depth
        )

        if len(cluster_nodes) >= min_cluster_size and current_depth < max_depth:
            sub_clusters = recursive_infomap(
                cluster_nodes, edges, node_lookup,
                min_cluster_size, max_depth,
                current_depth + 1,
                f"{cluster_id}."
            )
            if len(sub_clusters) > 1:
                cluster.children = sub_clusters
                cluster.nodes = []

        result.append(cluster)

    return result


def get_cluster_mean_layer(cluster: HierarchicalCluster) -> float:
    """Get mean layer of all nodes in cluster."""
    nodes = cluster.get_all_nodes()
    layers = [int(n.get('layer', 0)) for n in nodes if str(n.get('layer', '')).isdigit()]
    return sum(layers) / len(layers) if layers else 0


def get_leaf_clusters(clusters: list[HierarchicalCluster]) -> list[HierarchicalCluster]:
    """Recursively extract all leaf clusters."""
    leaves = []
    for cluster in clusters:
        if cluster.children:
            leaves.extend(get_leaf_clusters(cluster.children))
        else:
            leaves.append(cluster)
    return leaves


def flatten_clusters_sorted_by_layer(
    clusters: list[HierarchicalCluster]
) -> list[tuple[int, list[dict], float]]:
    """Flatten clusters to leaves sorted by mean layer."""
    leaves = get_leaf_clusters(clusters)

    cluster_data = []
    for cluster in leaves:
        nodes = cluster.nodes
        if not nodes:
            continue
        mean_layer = get_cluster_mean_layer(cluster)
        cluster_data.append((nodes, mean_layer))

    cluster_data.sort(key=lambda x: x[1])

    return [(i, nodes, mean_layer) for i, (nodes, mean_layer) in enumerate(cluster_data)]


def parse_logit_nodes(logit_nodes: list[dict]) -> list[dict[str, Any]]:
    """Parse logit nodes to extract token and probability."""
    top_logits = []
    for node in logit_nodes:
        clerp = node.get('clerp', '')
        match = re.match(r'\s*(.+?)\s*\(p=([0-9.]+)\)', clerp)
        if match:
            top_logits.append({
                'token': match.group(1),
                'probability': float(match.group(2))
            })
        else:
            top_logits.append({
                'token': clerp.strip(),
                'probability': 0
            })
    return top_logits


def cluster_graph(
    graph_data,
    skip_special_tokens: bool = True,
    min_cluster_size: int = 10,
    max_depth: int = 3,
    verbose: bool = True
) -> dict[str, Any]:
    """Cluster neurons in a graph using Infomap.

    Args:
        graph_data: Attribution graph - either a legacy dict or a Graph schema object
        skip_special_tokens: Filter out special token positions
        min_cluster_size: Min cluster size to subdivide
        max_depth: Max recursion depth
        verbose: Print progress

    Returns:
        Clustering results dict with clusters and top_logits
    """
    # If it's a Graph object, convert to legacy dict for existing code
    if isinstance(graph_data, Graph):
        graph_data = graph_data.to_legacy()
    skip_positions = set()
    if skip_special_tokens:
        skip_positions = get_special_token_positions(graph_data)
        if verbose and skip_positions:
            print(f"Skipping {len(skip_positions)} special token positions")

    neurons, node_lookup, logit_nodes = extract_neurons_and_logits(
        graph_data, skip_positions
    )

    if verbose:
        print(f"Found {len(neurons)} neurons for clustering")

    if len(neurons) < 3:
        raise ValueError("Too few neurons for clustering")

    edges = graph_data.get('links', [])

    if verbose:
        print(f"Running recursive Infomap (min_cluster={min_cluster_size}, max_depth={max_depth})...")

    clusters = recursive_infomap(
        neurons, edges, node_lookup,
        min_cluster_size=min_cluster_size,
        max_depth=max_depth
    )

    if verbose:
        print(f"Found {len(clusters)} top-level clusters")

    # Get metadata
    metadata = graph_data.get('metadata', {})
    prompt = metadata.get('prompt', '')
    prompt_tokens = metadata.get('prompt_tokens', [])

    def get_token(node: dict) -> str:
        token = node.get('ctx_token') or node.get('token')
        if token:
            return token
        ctx_idx = node.get('ctx_idx', node.get('position'))
        if ctx_idx is not None and prompt_tokens and 0 <= ctx_idx < len(prompt_tokens):
            token = prompt_tokens[ctx_idx]
            if token.startswith('Ġ') or token.startswith('\u0120'):
                token = token[1:]
            if token.startswith('Ċ') or token.startswith('\u010a'):
                token = '\\n'
            return token
        return ''

    # Flatten and sort clusters
    flat_clusters = flatten_clusters_sorted_by_layer(clusters)

    # Build cluster list
    cluster_list = []
    for cluster_id, nodes, mean_layer in flat_clusters:
        members = []
        for node in nodes:
            layer = node.get('layer', '')
            if isinstance(layer, int):
                layer = str(layer)

            members.append({
                'node_id': node['node_id'],
                'layer': layer,
                'neuron': node.get('feature', node.get('neuron', 0)),
                'position': node.get('position', node.get('ctx_idx', 0)),
                'token': get_token(node),
                'label': node.get('clerp', ''),
                'influence': node.get('influence', 0) or 0,
                'activation': node.get('activation', 0) or 0,
            })

        cluster_list.append({
            'cluster_id': cluster_id,
            'members': members
        })

    # Parse logits
    top_logits = parse_logit_nodes(logit_nodes)

    return {
        'prompt': prompt,
        'top_logits': top_logits,
        'methods': [{
            'method': 'infomap',
            'n_clusters': len(cluster_list),
            'clusters': cluster_list
        }]
    }


def cluster_full_model(
    edges: list[Edge],
    units: list[Unit],
    min_edge_count: int = 5,
    weight_transform: str = "abs_weight_sq",
    num_trials: int = 20,
    multi_level: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Infomap clustering on aggregated edge statistics for a full model.

    This is for clustering ALL neurons in a model (e.g., 1.6M neurons from
    800K RelP graphs), not for single-graph clustering.

    Args:
        edges: Aggregated edges (from InMemoryAggregator.get_edges())
        units: All observed units
        min_edge_count: Minimum edge observation count to include
        weight_transform: How to weight edges:
            "abs_weight" -> mean_abs_weight
            "abs_weight_sq" -> mean_abs_weight ** 2
            "weight" -> mean_weight (signed)
        num_trials: Number of Infomap optimization trials
        multi_level: Use multi-level (hierarchical) vs two-level
        verbose: Print progress

    Returns:
        Dict with:
        - assignments: {neuron_id_str: {"path": "1.5.3", "top": 1, "sub": 5, ...}}
        - stats: {total_clusters, singletons, gt2, gt5, gt10, ...}
        - top_clusters: [{id, size, layer_range, median_layer}]
    """
    if not HAS_INFOMAP:
        raise ImportError("infomap not installed. Run: uv pip install infomap")

    if verbose:
        print(f"cluster_full_model: {len(edges):,} edges, {len(units):,} units")

    # Build set of known unit keys for fast lookup
    known_units = {(u.layer, u.index) for u in units}

    # Step 1: Filter edges by min_edge_count and count edges per neuron
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    filtered_edges: list[Edge] = []

    for edge in edges:
        if edge.count < min_edge_count:
            continue
        src_key = (edge.src_layer, edge.src_index)
        tgt_key = (edge.tgt_layer, edge.tgt_index)
        if src_key not in known_units or tgt_key not in known_units:
            continue
        filtered_edges.append(edge)
        edge_count[src_key] += 1
        edge_count[tgt_key] += 1

    if verbose:
        print(f"  Edges after count>={min_edge_count} filter: {len(filtered_edges):,}")
        print(f"  Neurons with edges: {len(edge_count):,}")

    # Step 2: Keep only neurons with >1 edge (singletons can't cluster)
    valid_neurons = {k for k, c in edge_count.items() if c > 1}
    if verbose:
        print(f"  Neurons with >1 edge: {len(valid_neurons):,}")

    if len(valid_neurons) < 2:
        return {"assignments": {}, "stats": {}, "top_clusters": []}

    # Step 3: Map neuron keys to sequential integer IDs
    neuron_list = sorted(valid_neurons)
    neuron_to_idx = {n: i for i, n in enumerate(neuron_list)}
    N = len(neuron_list)

    # Step 4: Build Infomap network
    flags = "--directed"
    if multi_level:
        # Default multi-level (no --two-level flag)
        pass
    else:
        flags += " --two-level"
    flags += f" --num-trials {num_trials} --seed 42"

    if verbose:
        print(f"  Building Infomap network ({flags})...")

    im = infomap.Infomap(flags)
    n_added = 0

    for edge in filtered_edges:
        src_key = (edge.src_layer, edge.src_index)
        tgt_key = (edge.tgt_layer, edge.tgt_index)
        if src_key not in neuron_to_idx or tgt_key not in neuron_to_idx:
            continue

        # Compute weight based on transform
        if weight_transform == "abs_weight_sq":
            w = edge.mean_abs_weight ** 2
        elif weight_transform == "abs_weight":
            w = edge.mean_abs_weight
        elif weight_transform == "weight":
            w = abs(edge.mean_weight)
        else:
            raise ValueError(f"Unknown weight_transform: {weight_transform}")

        if w > 0:
            im.add_link(neuron_to_idx[src_key], neuron_to_idx[tgt_key], w)
            n_added += 1

    if verbose:
        print(f"  Added {n_added:,} directed edges to Infomap")

    if n_added == 0:
        return {"assignments": {}, "stats": {}, "top_clusters": []}

    # Step 5: Run Infomap
    if verbose:
        print("  Running Infomap...")

    im.run()

    if verbose:
        print(f"  Codelength: {im.codelength:.4f}")
        print(f"  Num top modules: {im.num_top_modules}")

    # Step 6: Extract hierarchical assignments
    assignments: dict[str, dict[str, Any]] = {}
    max_depth = 0

    for node in im.nodes:
        idx = node.node_id
        if idx >= N:
            continue
        layer, index = neuron_list[idx]
        neuron_id = f"L{layer}/N{index}"
        path = node.path
        depth = len(path)
        max_depth = max(max_depth, depth)

        assignments[neuron_id] = {
            "path": ".".join(str(p) for p in path),
            "top": path[0] if len(path) > 0 else -1,
            "sub": path[1] if len(path) > 1 else None,
            "subsub": path[2] if len(path) > 2 else None,
            "depth": depth,
            "flow": float(node.flow),
        }

    if verbose:
        print(f"  Assignments: {len(assignments):,} neurons")
        print(f"  Max hierarchy depth: {max_depth}")

    # Step 7: Compute cluster statistics
    top_cluster_members: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for neuron_id_str, info in assignments.items():
        # Parse "L{layer}/N{index}" back to layer, index
        parts = neuron_id_str.split("/")
        layer = int(parts[0][1:])
        index = int(parts[1][1:])
        top_cluster_members[info["top"]].append((layer, index))

    sizes = [len(v) for v in top_cluster_members.values()]
    max_size = max(sizes) if sizes else 0

    stats = {
        "total_clusters": len(top_cluster_members),
        "singletons": sum(1 for s in sizes if s == 1),
        "gt2": sum(1 for s in sizes if s > 2),
        "gt5": sum(1 for s in sizes if s > 5),
        "gt10": sum(1 for s in sizes if s > 10),
        "gt50": sum(1 for s in sizes if s > 50),
        "gt100": sum(1 for s in sizes if s > 100),
        "gt1000": sum(1 for s in sizes if s > 1000),
        "max_cluster": max_size,
        "max_cluster_pct": round(100 * max_size / len(assignments), 1) if assignments else 0,
        "codelength": im.codelength,
        "num_top_modules": im.num_top_modules,
        "max_depth": max_depth,
        "total_neurons": len(assignments),
        "total_edges": n_added,
    }

    if verbose:
        print(f"  Top-level clusters: {stats['total_clusters']:,}")
        print(f"  Clusters >100: {stats['gt100']}")
        print(f"  Max cluster: {max_size:,}")

    # Build top_clusters summary (sorted by size descending)
    sorted_clusters = sorted(
        top_cluster_members.items(), key=lambda x: -len(x[1])
    )

    def _median(values: list[int]) -> float:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return float(s[mid])

    top_clusters_list = []
    for cid, members in sorted_clusters[:200]:
        layers = [layer for layer, _ in members]
        top_clusters_list.append({
            "id": cid,
            "size": len(members),
            "layer_range": [min(layers), max(layers)],
            "median_layer": _median(layers),
        })

    return {
        "assignments": assignments,
        "stats": stats,
        "top_clusters": top_clusters_list,
    }
