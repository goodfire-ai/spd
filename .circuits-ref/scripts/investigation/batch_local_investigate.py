#!/usr/bin/env python3
"""Batch neuron investigation locally with parallel agents.

Runs multiple investigations in parallel on the current GPU.

Usage:
    # Investigate neurons from a circuit graph
    python scripts/batch_local_investigate.py \
        --graph graphs/aspirin-cox-target.json \
        --parallel 4

    # Limit to first N neurons
    python scripts/batch_local_investigate.py \
        --graph graphs/aspirin-cox-target.json \
        --parallel 4 \
        --limit 20
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron_scientist import investigate_neuron


def get_neurons_from_graph(graph_path: str, v6_only: bool = True) -> list:
    """Extract neurons from a graph file."""
    with open(graph_path) as f:
        graph = json.load(f)

    v6_neurons = set()
    if v6_only:
        v6_path = Path("data/medical_edge_stats_v6_enriched.json")
        if v6_path.exists():
            with open(v6_path) as f:
                v6_data = json.load(f)
            v6_neurons = set((p['layer'], p['neuron']) for p in v6_data['profiles'])

    investigated = set()
    inv_dir = Path("outputs/investigations")
    if inv_dir.exists():
        for f in inv_dir.glob("*_dashboard.json"):
            parts = f.stem.replace("_dashboard", "").split("_")
            if len(parts) == 2:
                try:
                    layer = int(parts[0][1:])
                    neuron = int(parts[1][1:])
                    investigated.add((layer, neuron))
                except:
                    pass

    neurons = []
    for node in graph.get('nodes', []):
        if node.get('feature_type') == 'mlp_neuron':
            try:
                layer = int(node['layer'])
                neuron = node['feature']
                influence = node.get('influence') or 0

                if (layer, neuron) in investigated:
                    continue
                if v6_only and v6_neurons and (layer, neuron) not in v6_neurons:
                    continue

                neurons.append({
                    'layer': layer,
                    'neuron': neuron,
                    'influence': influence
                })
            except:
                pass

    neurons.sort(key=lambda x: (-x['layer'], -x['influence']))
    return neurons


async def investigate_one(layer: int, neuron: int, output_dir: str,
                          edge_stats: str = None, labels_path: str = None,
                          idx: int = 0, total: int = 0):
    """Investigate a single neuron."""
    neuron_id = f"L{layer}/N{neuron}"
    print(f"[{idx+1}/{total}] Starting: {neuron_id}", flush=True)

    try:
        result = await investigate_neuron(
            neuron_id=neuron_id,
            initial_label="",
            initial_hypothesis="",
            edge_stats_path=edge_stats,
            labels_path=Path(labels_path) if labels_path else None,
            output_dir=output_dir,
            max_experiments=50,
        )
        print(f"[{idx+1}/{total}] ✓ Completed: {neuron_id} (confidence={result.confidence:.0%}, "
              f"experiments={result.total_experiments})", flush=True)
        return True
    except Exception as e:
        print(f"[{idx+1}/{total}] ✗ FAILED: {neuron_id} - {e}", flush=True)
        return False


async def main_async(args):
    """Async main function."""
    # Get neurons
    if args.graph:
        neurons = get_neurons_from_graph(args.graph, v6_only=not args.no_v6_filter)
    else:
        print("Error: Must specify --graph")
        return

    if not neurons:
        print("No neurons to investigate!")
        return

    if args.limit:
        neurons = neurons[:args.limit]

    total = len(neurons)
    print("=" * 60)
    print("LOCAL BATCH INVESTIGATION")
    print("=" * 60)
    print(f"Neurons to investigate: {total}")
    print(f"Parallel agents: {args.parallel}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Show first few
    print("\nNeurons (late to early):")
    for n in neurons[:10]:
        print(f"  L{n['layer']}/N{n['neuron']} (influence={n['influence']:.3f})")
    if len(neurons) > 10:
        print(f"  ... and {len(neurons) - 10} more")
    print()

    # Run with semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.parallel)
    completed = 0
    failed = 0

    async def bounded_investigate(n, idx):
        nonlocal completed, failed
        async with semaphore:
            success = await investigate_one(
                n['layer'], n['neuron'],
                args.output_dir, args.edge_stats, args.labels,
                idx, total
            )
            if success:
                completed += 1
            else:
                failed += 1

    start_time = datetime.now()
    tasks = [bounded_investigate(n, i) for i, n in enumerate(neurons)]
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"Completed: {completed}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/total:.1f}s per neuron)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch neuron investigation (local)")
    parser.add_argument("--graph", required=True, help="Graph file to extract neurons from")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel agents")
    parser.add_argument("--limit", type=int, help="Limit to first N neurons")
    parser.add_argument("--edge-stats", help="Edge stats file")
    parser.add_argument("--labels", default="data/neuron_labels_combined.json",
                       help="Neuron labels file (default: combined FineWeb + v6)")
    parser.add_argument("--output-dir", default="neuron_reports/json", help="Output directory")
    parser.add_argument("--no-v6-filter", action="store_true", help="Include non-v6 neurons")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
