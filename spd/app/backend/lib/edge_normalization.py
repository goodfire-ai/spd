"""Edge normalization utilities for attribution graphs."""

import copy
from collections import defaultdict

from spd.app.backend.compute import Edge


def normalize_edges_by_target(edges: list[Edge]) -> list[Edge]:
    """Normalize edges so incoming edges to each target node sum to 1.

    Groups edges by target node (target:s_out:c_out_idx) and normalizes
    the absolute values of incoming edges to sum to 1, preserving signs.

    Args:
        edges: List of Edge dataclasses.

    Returns:
        New list of edges with normalized values.
    """
    if not edges:
        return edges

    # Group edges by target node
    edges_by_target: dict[str, list[tuple[int, Edge]]] = defaultdict(list)
    for i, edge in enumerate(edges):
        edges_by_target[str(edge.target)].append((i, edge))

    # Normalize each group
    normalized = copy.copy(edges)  # Shallow copy of list
    for _target_key, edge_group in edges_by_target.items():
        # Sum of absolute values
        total_abs = sum(edge.strength**2 for _, edge in edge_group)
        if total_abs == 0:
            continue

        # Normalize: val -> val / total_abs (preserves sign)
        for idx, edge in edge_group:
            new_val = edge.strength / total_abs
            # Create new Edge with updated strength
            normalized[idx] = Edge(
                source=edge.source,
                target=edge.target,
                strength=new_val,
                is_cross_seq=edge.is_cross_seq,
            )

    return normalized
