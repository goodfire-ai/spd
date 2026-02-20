#!/usr/bin/env python3
"""Add output projections from edge stats to the neuron labels file.

The edge stats file contains pre-computed output projections (what tokens each neuron
promotes/suppresses via its down_proj @ lm_head weights). This script merges that data
into the combined labels file.

Usage:
    python scripts/add_output_projections.py \
        --labels data/neuron_labels_combined.json \
        --edge-stats data/fineweb_50k_edge_stats_enriched.json \
        -o data/neuron_labels_combined.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Add output projections to labels file")
    parser.add_argument("--labels", "-l", type=Path, required=True,
                        help="Input labels file (neuron_labels_combined.json)")
    parser.add_argument("--edge-stats", "-e", type=Path, required=True,
                        help="Edge stats file with output_projection data")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output labels file")
    args = parser.parse_args()

    # Load labels
    print(f"Loading labels from {args.labels}...")
    with open(args.labels) as f:
        labels_data = json.load(f)

    neurons = labels_data.get("neurons", {})
    print(f"  Found {len(neurons)} neurons")

    # Load edge stats
    print(f"Loading edge stats from {args.edge_stats}...")
    with open(args.edge_stats) as f:
        edge_stats = json.load(f)

    profiles = edge_stats.get("profiles", [])
    print(f"  Found {len(profiles)} profiles")

    # Build lookup by neuron_id
    edge_stats_by_id = {}
    for p in profiles:
        nid = p.get("neuron_id")
        if nid:
            edge_stats_by_id[nid] = p

    # Merge output projections
    updated_count = 0
    for neuron_id, neuron_data in neurons.items():
        if neuron_id in edge_stats_by_id:
            edge_profile = edge_stats_by_id[neuron_id]
            output_proj = edge_profile.get("output_projection", {})

            if output_proj:
                promoted = output_proj.get("promoted", [])
                suppressed = output_proj.get("suppressed", [])

                # Store in direct_logit_effects with standardized format
                neuron_data["direct_logit_effects"] = {
                    "promotes": [
                        {"token": t.get("token", ""), "weight": t.get("weight", 0)}
                        for t in promoted[:20]
                    ],
                    "suppresses": [
                        {"token": t.get("token", ""), "weight": t.get("weight", 0)}
                        for t in suppressed[:20]
                    ],
                }
                updated_count += 1

    print(f"Updated {updated_count} neurons with output projections")

    # Save
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(labels_data, f)

    print("Done!")

    # Verify a sample
    sample_id = "L17/N12426"
    if sample_id in neurons:
        dle = neurons[sample_id].get("direct_logit_effects", {})
        promotes = dle.get("promotes", [])
        if promotes:
            print(f"\nSample {sample_id} promotes:")
            for t in promotes[:5]:
                print(f"  {t['token']}: {t['weight']:.4f}")


if __name__ == "__main__":
    main()
