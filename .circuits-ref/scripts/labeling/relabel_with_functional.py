#!/usr/bin/env python3
"""
Relabel attribution graphs with functional descriptions from the progressive interpretation database.

Replaces the NeuronDB-based 'clerp' labels with our computed function_label and input_description.
"""

import argparse
import json
from pathlib import Path


def load_functional_db(db_path: str) -> dict:
    """Load the neuron function database."""
    with open(db_path) as f:
        data = json.load(f)
    return data.get('neurons', {})


def relabel_graph(graph_path: str, func_db: dict, output_path: str = None):
    """
    Relabel a graph with functional descriptions.

    Args:
        graph_path: Path to input graph JSON
        func_db: Dictionary of neuron_id -> neuron data
        output_path: Path to save relabeled graph (default: adds _functional suffix)
    """
    with open(graph_path) as f:
        graph = json.load(f)

    if output_path is None:
        p = Path(graph_path)
        output_path = str(p.parent / f"{p.stem}_functional{p.suffix}")

    labeled_count = 0
    total_mlp = 0

    for node in graph['nodes']:
        if node.get('feature_type') != 'mlp_neuron':
            continue

        total_mlp += 1

        # Parse node_id: "1_2427_0" -> layer=1, neuron=2427
        parts = node['node_id'].split('_')
        layer = int(parts[0])
        neuron = int(parts[1])
        neuron_id = f"L{layer}/N{neuron}"

        # Look up in functional database
        if neuron_id in func_db:
            ndata = func_db[neuron_id]
            func_label = ndata.get('function_label', '')
            input_desc = ndata.get('input_description', '')

            # Create new clerp combining both
            if func_label and input_desc:
                new_clerp = f"{neuron_id}: {func_label} | fires: {input_desc}"
            elif func_label:
                new_clerp = f"{neuron_id}: {func_label}"
            elif input_desc:
                new_clerp = f"{neuron_id}: fires: {input_desc}"
            else:
                new_clerp = neuron_id

            # Store both as separate fields too
            node['clerp'] = new_clerp
            node['function_label'] = func_label
            node['input_description'] = input_desc
            labeled_count += 1
        else:
            # Keep original clerp but note it's unlabeled
            node['function_label'] = ''
            node['input_description'] = ''

    # Add metadata about relabeling
    if 'metadata' not in graph:
        graph['metadata'] = {}
    graph['metadata']['relabeled_with'] = 'progressive_interpretation_functional_db'
    graph['metadata']['functional_labels'] = labeled_count
    graph['metadata']['total_mlp_neurons'] = total_mlp

    # Save
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2)

    print(f"Relabeled {labeled_count}/{total_mlp} neurons in {Path(graph_path).name}")
    print(f"  Saved to: {output_path}")

    return output_path


def relabel_clusters(clusters_path: str, func_db: dict, output_path: str = None):
    """Relabel cluster file with functional descriptions."""
    with open(clusters_path) as f:
        clusters = json.load(f)

    if output_path is None:
        p = Path(clusters_path)
        output_path = str(p.parent / f"{p.stem}_functional{p.suffix}")

    # Relabel neurons in cluster data
    for cluster in clusters.get('clusters', []):
        for neuron in cluster.get('neurons', []):
            if 'neuron_id' in neuron:
                nid = neuron['neuron_id']
                if nid in func_db:
                    ndata = func_db[nid]
                    neuron['function_label'] = ndata.get('function_label', '')
                    neuron['input_description'] = ndata.get('input_description', '')
                    if neuron['function_label']:
                        neuron['label'] = f"{nid}: {neuron['function_label']}"

    with open(output_path, 'w') as f:
        json.dump(clusters, f, indent=2)

    print(f"  Saved clusters to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Relabel graphs with functional descriptions')
    parser.add_argument('--db', default='data/neuron_function_db_full.json',
                       help='Path to neuron function database')
    parser.add_argument('--graphs', nargs='+', required=True,
                       help='Graph files to relabel')
    parser.add_argument('--output-dir', help='Output directory (default: same as input)')

    args = parser.parse_args()

    print(f"Loading functional database from {args.db}...")
    func_db = load_functional_db(args.db)
    print(f"  Loaded {len(func_db)} neuron descriptions")
    print()

    for graph_path in args.graphs:
        if args.output_dir:
            p = Path(graph_path)
            output_path = str(Path(args.output_dir) / f"{p.stem}_functional{p.suffix}")
        else:
            output_path = None

        relabel_graph(graph_path, func_db, output_path)

        # Also relabel clusters if they exist
        clusters_path = graph_path.replace('-graph.json', '-clusters.json')
        if Path(clusters_path).exists():
            if args.output_dir:
                p = Path(clusters_path)
                clusters_output = str(Path(args.output_dir) / f"{p.stem}_functional{p.suffix}")
            else:
                clusters_output = None
            relabel_clusters(clusters_path, func_db, clusters_output)
        print()


if __name__ == '__main__':
    main()
