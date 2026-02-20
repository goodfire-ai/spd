#!/usr/bin/env python3
"""
Merge DER batch results and add to v6_enriched file.

Usage:
    python scripts/merge_der_results.py
"""

import json
from glob import glob


def main():
    # Find all batch result files
    batch_files = sorted(glob('outputs/der_jobs/der_batch_*.json'))
    print(f"Found {len(batch_files)} batch files")

    # Merge results
    all_results = {}
    for path in batch_files:
        print(f"  Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        all_results.update(data['results'])

    print(f"Total DER results: {len(all_results)}")

    # Load v6 enriched
    print("\nLoading v6_enriched...")
    with open('data/medical_edge_stats_v6_enriched.json') as f:
        v6 = json.load(f)

    # Add/update DER to profiles
    added = 0
    updated = 0
    for profile in v6['profiles']:
        neuron_id = profile['neuron_id']
        if neuron_id in all_results:
            if 'direct_effect_ratio' not in profile:
                added += 1
            else:
                updated += 1
            profile['direct_effect_ratio'] = all_results[neuron_id]

    print(f"Added DER to {added} profiles, updated {updated} profiles")

    # Update metadata
    der_count = sum(1 for p in v6['profiles'] if 'direct_effect_ratio' in p)
    v6['metadata']['enrichments']['direct_effect_ratio_count'] = der_count
    v6['metadata']['enrichments']['der_computed_for_missing'] = added

    # Save
    output_path = 'data/medical_edge_stats_v6_enriched.json'
    with open(output_path, 'w') as f:
        json.dump(v6, f)

    print(f"\nSaved to {output_path}")
    print(f"Total profiles with DER: {der_count}/{len(v6['profiles'])}")


if __name__ == '__main__':
    main()
