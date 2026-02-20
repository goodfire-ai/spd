"""Neuron labeling from database.

Fetches neuron descriptions from PostgreSQL database and applies them to graphs.
This module is a first-class citizen of the pipeline - use label_graph() directly.

The database connection uses the observatory_repo/lib/neurondb package, which
requires the PostgreSQL neurondb to be running (see CLAUDE.md for setup).
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Cache for database manager to avoid repeated imports
_db_manager = None
_db_available = None


@dataclass
class NeuronInfo:
    """Info extracted from attribution graph node."""
    layer: int
    neuron: int
    node_id: str
    polarity: int = 1  # 1 for positive activation, -1 for negative


def get_observatory_path() -> Path | None:
    """Get path to observatory_repo directory."""
    # Try relative to this file first
    path = Path(__file__).parent.parent / "observatory_repo"
    if path.exists():
        return path

    # Try relative to cwd
    path = Path("observatory_repo")
    if path.exists():
        return path.absolute()

    return None


@contextmanager
def observatory_context():
    """Context manager for safely importing from observatory_repo.

    Temporarily modifies sys.path and cwd, then restores them.
    """
    observatory_path = get_observatory_path()
    if observatory_path is None:
        raise ImportError("observatory_repo not found")

    original_cwd = os.getcwd()
    original_path = sys.path.copy()

    try:
        os.chdir(observatory_path)

        # Add lib subdirectories to path
        lib_path = observatory_path / "lib"
        if lib_path.exists():
            for subdir in lib_path.iterdir():
                if subdir.is_dir():
                    sys.path.insert(0, str(subdir))

        yield observatory_path

    finally:
        os.chdir(original_cwd)
        sys.path = original_path


def get_db_manager():
    """Get or create database manager instance.

    Returns None if database is unavailable.
    """
    global _db_manager, _db_available

    if _db_available is False:
        return None

    if _db_manager is not None:
        return _db_manager

    try:
        with observatory_context():
            from neurondb.postgres import DBManager
            _db_manager = DBManager.get_instance()
            _db_available = True
            return _db_manager
    except Exception:
        _db_available = False
        return None


def is_database_available() -> bool:
    """Check if the neuron database is available."""
    global _db_available

    if _db_available is not None:
        return _db_available

    try:
        db = get_db_manager()
        return db is not None
    except Exception:
        _db_available = False
        return False


def extract_neurons_from_graph(graph: dict) -> list[NeuronInfo]:
    """Extract MLP neuron info from attribution graph."""
    neurons = []
    for node in graph.get("nodes", []):
        if node.get("feature_type") == "mlp_neuron":
            layer = node.get("layer")
            neuron = node.get("feature")
            node_id = node.get("node_id")

            if layer is not None and neuron is not None:
                if isinstance(layer, str) and layer.isdigit():
                    layer = int(layer)
                elif isinstance(layer, str):
                    continue

                activation = node.get("activation", 0) or 0
                polarity = 1 if activation >= 0 else -1
                neurons.append(NeuronInfo(
                    layer=layer,
                    neuron=neuron,
                    node_id=node_id,
                    polarity=polarity
                ))
    return neurons


def fetch_descriptions(
    neurons: list[NeuronInfo],
    timeout_ms: int = 60000
) -> dict[str, str]:
    """Fetch neuron descriptions from the PostgreSQL database.

    Args:
        neurons: List of neurons to fetch descriptions for
        timeout_ms: Database query timeout in milliseconds

    Returns:
        Dict mapping node_id to description
    """
    if not neurons:
        return {}

    try:
        with observatory_context():
            from neurondb.postgres import DBManager
            from neurondb.schemas import SQLANeuron, SQLANeuronDescription

            db = DBManager.get_instance()

            layer_neuron_tuples = [(n.layer, n.neuron) for n in neurons]

            results = db.get(
                [SQLANeuron.layer, SQLANeuron.neuron,
                 SQLANeuronDescription.description, SQLANeuronDescription.polarity],
                joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
                layer_neuron_tuples=layer_neuron_tuples,
                timeout_ms=timeout_ms
            )

            # Build lookup: (layer, neuron, polarity) -> description
            desc_lookup: dict[tuple[int, int, int], str] = {}
            for row in results:
                layer, neuron, description, polarity = row
                desc_lookup[(layer, neuron, polarity)] = description

            # Map node_ids to descriptions
            descriptions: dict[str, str] = {}
            for n in neurons:
                # Try exact polarity match first
                key = (n.layer, n.neuron, n.polarity)
                if key in desc_lookup:
                    descriptions[n.node_id] = desc_lookup[key]
                else:
                    # Fall back to positive polarity
                    key_positive = (n.layer, n.neuron, 1)
                    if key_positive in desc_lookup:
                        descriptions[n.node_id] = desc_lookup[key_positive]

            return descriptions

    except Exception as e:
        import traceback
        print(f"Error fetching from database: {e}", file=sys.stderr)
        traceback.print_exc()
        return {}


def apply_descriptions_to_graph(
    graph: dict,
    descriptions: dict[str, str],
) -> tuple[dict, int, int]:
    """Update graph nodes with fetched descriptions.

    Args:
        graph: Attribution graph dict
        descriptions: Dict mapping node_id to description

    Returns:
        Tuple of (updated_graph, nodes_updated, features_updated)
    """
    nodes_updated = 0
    features_updated = 0

    # Update nodes
    for node in graph.get("nodes", []):
        if node.get("feature_type") == "mlp_neuron":
            node_id = node.get("node_id")
            if node_id in descriptions:
                desc = descriptions[node_id]
                layer = node.get("layer")
                neuron = node.get("feature")
                node["clerp"] = f"L{layer}/N{neuron}: {desc}"
                nodes_updated += 1

    # Update features array (for viewer compatibility)
    for feature in graph.get("features", []):
        if feature.get("feature_type") == "mlp_neuron":
            layer = feature.get("layer")
            feature_idx = feature.get("featureIndex")
            if layer is not None and feature_idx is not None:
                for node_id, desc in descriptions.items():
                    parts = node_id.split("_")
                    if len(parts) >= 2:
                        n_layer, n_neuron = parts[0], parts[1]
                        if str(layer) == n_layer and str(feature_idx) == n_neuron:
                            feature["clerp"] = f"L{layer}/N{feature_idx}: {desc}"
                            features_updated += 1
                            break

    return graph, nodes_updated, features_updated


def label_graph(
    graph: dict,
    verbose: bool = True,
    timeout_ms: int = 60000
) -> dict:
    """Add neuron labels from database to graph.

    This is the main entry point for labeling. Call this directly
    from the pipeline - no subprocess needed.

    Args:
        graph: Attribution graph dict
        verbose: Print progress messages
        timeout_ms: Database query timeout

    Returns:
        Graph with updated neuron labels
    """
    neurons = extract_neurons_from_graph(graph)

    if not neurons:
        if verbose:
            print("No MLP neurons found in graph")
        return graph

    if verbose:
        print(f"Found {len(neurons)} MLP neurons to label")

    # Check if database is available
    if not is_database_available():
        if verbose:
            print("Warning: Neuron database not available, skipping labeling")
        return graph

    descriptions = fetch_descriptions(neurons, timeout_ms=timeout_ms)

    if verbose:
        print(f"Fetched {len(descriptions)} descriptions from database")

    graph, nodes_updated, features_updated = apply_descriptions_to_graph(graph, descriptions)

    if verbose:
        print(f"  Updated {nodes_updated} neuron labels")
        if features_updated > 0:
            print(f"  Updated {features_updated} feature labels")

    return graph


# ============================================================================
# Batch operations for multiple graphs
# ============================================================================

def label_graphs(
    graphs: list[dict],
    verbose: bool = True
) -> list[dict]:
    """Label multiple graphs efficiently.

    Collects all neurons first, fetches descriptions in one batch,
    then applies to all graphs.

    Args:
        graphs: List of attribution graph dicts
        verbose: Print progress

    Returns:
        List of graphs with updated labels
    """
    if not graphs:
        return []

    # Collect all neurons from all graphs
    all_neurons = []
    graph_neurons = []  # Track which neurons belong to which graph

    for i, graph in enumerate(graphs):
        neurons = extract_neurons_from_graph(graph)
        all_neurons.extend(neurons)
        graph_neurons.append(neurons)

    if not all_neurons:
        if verbose:
            print("No MLP neurons found in any graph")
        return graphs

    if verbose:
        print(f"Found {len(all_neurons)} total neurons across {len(graphs)} graphs")

    # Fetch all descriptions at once
    if not is_database_available():
        if verbose:
            print("Warning: Neuron database not available, skipping labeling")
        return graphs

    descriptions = fetch_descriptions(all_neurons)

    if verbose:
        print(f"Fetched {len(descriptions)} descriptions")

    # Apply to each graph
    result_graphs = []
    for graph in graphs:
        updated_graph, _, _ = apply_descriptions_to_graph(graph, descriptions)
        result_graphs.append(updated_graph)

    return result_graphs
