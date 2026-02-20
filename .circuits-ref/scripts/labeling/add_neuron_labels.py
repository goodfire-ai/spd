#!/usr/bin/env python3
"""
Add NeuronDB autointerp labels to edge aggregation data.

Takes edge stats JSON from parallel_edge_aggregation.py and adds "max_act_label"
field to each neuron profile from the NeuronDB database.

Usage:
    python scripts/add_neuron_labels.py data/medical_edge_stats_1000.json -o data/medical_edge_stats_labeled.json

    # Dry run - just show sample labels
    python scripts/add_neuron_labels.py data/medical_edge_stats_1000.json --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_neurondb():
    """Initialize NeuronDB connection."""
    # Change to observatory_repo for imports
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root / "observatory_repo")

    from dotenv import load_dotenv
    load_dotenv()

    from neurondb.postgres import DBManager
    from neurondb.schemas import SQLANeuron, SQLANeuronDescription

    db = DBManager.get_instance()
    return db, SQLANeuron, SQLANeuronDescription


def get_neuron_labels(
    db,
    SQLANeuron,
    SQLANeuronDescription,
    neuron_tuples: list[tuple[int, int]],
    batch_size: int = 500,
) -> dict[tuple[int, int], str]:
    """
    Query NeuronDB for labels for a batch of neurons.

    Args:
        db: DBManager instance
        SQLANeuron, SQLANeuronDescription: Schema classes
        neuron_tuples: List of (layer, neuron) tuples
        batch_size: Batch size for database queries

    Returns:
        Dict mapping (layer, neuron) -> description
    """
    labels = {}

    # Process in batches
    for i in range(0, len(neuron_tuples), batch_size):
        batch = neuron_tuples[i:i + batch_size]

        try:
            results = db.get(
                [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
                joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
                layer_neuron_tuples=batch
            )

            for layer, neuron, desc in results:
                labels[(layer, neuron)] = desc

        except Exception as e:
            print(f"Warning: Error querying batch {i//batch_size}: {e}", file=sys.stderr)

    return labels


def add_labels_to_edge_stats(
    edge_stats_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Add NeuronDB labels to edge stats JSON.

    Args:
        edge_stats_path: Path to input JSON from parallel_edge_aggregation.py
        output_path: Path for output JSON (if None, modifies in place)
        dry_run: If True, just print sample labels without saving

    Returns:
        Modified edge stats dict
    """
    # Resolve paths before changing directory
    edge_stats_path = Path(edge_stats_path).resolve()
    if output_path:
        output_path = Path(output_path).resolve()

    # Load edge stats
    print(f"Loading edge stats from {edge_stats_path}...")
    with open(edge_stats_path) as f:
        data = json.load(f)

    profiles = data.get("profiles", [])
    print(f"Found {len(profiles)} neuron profiles")

    # Extract (layer, neuron) tuples
    neuron_tuples = []
    for p in profiles:
        layer = p.get("layer")
        neuron = p.get("neuron")
        if layer is not None and neuron is not None:
            neuron_tuples.append((layer, neuron))

    print(f"Querying NeuronDB for {len(neuron_tuples)} neurons...")

    # Connect to NeuronDB
    db, SQLANeuron, SQLANeuronDescription = load_neurondb()

    # Get labels
    labels = get_neuron_labels(db, SQLANeuron, SQLANeuronDescription, neuron_tuples)
    print(f"Got labels for {len(labels)} neurons ({len(labels)/len(neuron_tuples)*100:.1f}%)")

    if dry_run:
        # Show sample labels
        print("\nSample labels:")
        for i, p in enumerate(profiles[:10]):
            layer, neuron = p["layer"], p["neuron"]
            label = labels.get((layer, neuron), "NO LABEL FOUND")
            # Truncate long labels
            if len(label) > 100:
                label = label[:100] + "..."
            print(f"  L{layer}/N{neuron}: {label}")
        return data

    # Add labels to profiles
    labeled_count = 0
    for p in profiles:
        layer, neuron = p["layer"], p["neuron"]
        label = labels.get((layer, neuron))
        if label:
            p["max_act_label"] = label
            labeled_count += 1
        else:
            p["max_act_label"] = None

    print(f"Added labels to {labeled_count}/{len(profiles)} profiles")

    # Update metadata
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["neurondb_labeled"] = True
    data["metadata"]["labeled_count"] = labeled_count

    # Save output
    if output_path is None:
        output_path = edge_stats_path

    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Done!")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Add NeuronDB autointerp labels to edge stats"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input edge stats JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (default: overwrite input)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show sample labels without saving"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    add_labels_to_edge_stats(
        args.input,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
