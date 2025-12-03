"""Edge normalization utilities for attribution graphs."""

from collections import defaultdict

from spd.app.backend.compute import Edge


def normalize_edges_by_target(edges: list[Edge]) -> list[Edge]:
    """Normalize edges so incoming edges to each target node sum to 1.

    Groups edges by target node (target:s_out:c_out_idx) and normalizes
    the absolute values of incoming edges to sum to 1, preserving signs.

    Args:
        edges: List of edge tuples.

    Returns:
        New list of edges with normalized values.
    """
    if not edges:
        return edges

    # Group edges by target node
    edges_by_target: dict[str, list[tuple[int, Edge]]] = defaultdict(list)
    for i, edge in enumerate(edges):
        # edge: (source, target, c_in, c_out, s_in, s_out, strength, is_cross_seq)
        _, target, _, c_out_idx, _, s_out, _, _ = edge
        target_key = f"{target}:{s_out}:{c_out_idx}"
        edges_by_target[target_key].append((i, edge))

    # Normalize each group
    normalized = list(edges)  # Copy
    for _target_key, edge_group in edges_by_target.items():
        # Sum of absolute values
        total_abs = sum(abs(edge[6]) for _, edge in edge_group)
        if total_abs == 0:
            continue

        # Normalize: val -> val / total_abs (preserves sign)
        for idx, edge in edge_group:
            old_val = edge[6]
            new_val = old_val / total_abs
            # Create new tuple with updated value
            normalized[idx] = (
                edge[0],
                edge[1],
                edge[2],
                edge[3],
                edge[4],
                edge[5],
                new_val,
                edge[7],
            )

    return normalized
