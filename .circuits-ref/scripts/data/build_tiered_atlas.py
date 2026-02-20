#!/usr/bin/env python3
"""Build a tiered circuit atlas combining RelP and weight-based clustering.

Tier 1: RelP Infomap (config 27: infomap_2lvl.relp.sq.q90.t20)
  - Take all non-mega-cluster assignments
  - Discard mega-cluster neurons back to the unassigned pool

Tier 2: Weight-based Infomap (config 53: infomap_2lvl.weight.sq.q90.t10)
  - Cluster the remaining pool (unassigned + mega-cluster discards)
  - Only assign neurons that weren't already assigned in Tier 1

Output: DuckDB database with unified atlas.

Usage:
    .venv/bin/python scripts/build_tiered_atlas.py
"""

import json
import os
import time
from collections import Counter

import duckdb


def main():
    print("Building Tiered Circuit Atlas")
    print("=" * 60)
    t0 = time.time()

    # ============================================================
    # LOAD TIER 1: RelP Infomap (config 27)
    # ============================================================
    print("\n[1/5] Loading Tier 1: RelP Infomap (config 27)...")

    relp_assigns = json.load(open("data/clustering_sweep/config_027_assignments.json"))
    relp_metrics = json.load(open("data/clustering_sweep/config_027_metrics.json"))

    # Find the mega-cluster (largest cluster)
    cluster_sizes = Counter(relp_assigns.values())
    mega_cluster_id = cluster_sizes.most_common(1)[0][0]
    mega_size = cluster_sizes[mega_cluster_id]

    print(f"  Total RelP assignments: {len(relp_assigns):,}")
    print(f"  Mega-cluster: id={mega_cluster_id}, size={mega_size:,} "
          f"({100*mega_size/len(relp_assigns):.1f}%)")
    print(f"  Non-mega clusters: {len(cluster_sizes) - 1:,}")

    # Tier 1: everything EXCEPT the mega-cluster
    tier1 = {}
    tier1_discarded = set()  # neurons in mega-cluster, returned to pool
    for node_str, cluster_id in relp_assigns.items():
        if cluster_id == mega_cluster_id:
            tier1_discarded.add(node_str)
        else:
            tier1[node_str] = ("relp", cluster_id)

    # Also track neurons with no RelP assignment at all
    # (the 75% "dark matter")

    tier1_clusters = Counter(cid for _, (_, cid) in tier1.items())
    print(f"  Tier 1 neurons: {len(tier1):,}")
    print(f"  Tier 1 clusters: {len(tier1_clusters):,}")
    print(f"  Discarded (mega-cluster): {len(tier1_discarded):,}")

    # ============================================================
    # LOAD TIER 2: Weight-based Infomap (config 53)
    # ============================================================
    print("\n[2/5] Loading Tier 2: Weight-based Infomap (config 53)...")

    weight_assigns = json.load(open("data/clustering_sweep/config_053_assignments.json"))
    weight_sizes = Counter(weight_assigns.values())
    weight_mega_id = weight_sizes.most_common(1)[0][0]
    weight_mega_size = weight_sizes[weight_mega_id]

    print(f"  Total weight assignments: {len(weight_assigns):,}")
    print(f"  Weight mega-cluster: id={weight_mega_id}, size={weight_mega_size:,}")

    # Tier 2: weight assignments for neurons NOT in Tier 1
    # Also discard the weight mega-cluster
    tier2 = {}
    tier2_skipped_already_assigned = 0
    tier2_skipped_mega = 0

    # Offset weight cluster IDs to avoid collision with RelP IDs
    max_relp_id = max(cid for _, cid in tier1.values()) if tier1 else 0
    weight_offset = max_relp_id + 1000  # safety gap

    for node_str, cluster_id in weight_assigns.items():
        if node_str in tier1:
            tier2_skipped_already_assigned += 1
            continue
        if cluster_id == weight_mega_id:
            tier2_skipped_mega += 1
            continue
        tier2[node_str] = ("weight", cluster_id + weight_offset)

    tier2_clusters = Counter(cid for _, (_, cid) in tier2.items())
    print(f"  Tier 2 neurons: {len(tier2):,}")
    print(f"  Tier 2 clusters: {len(tier2_clusters):,}")
    print(f"  Skipped (already in Tier 1): {tier2_skipped_already_assigned:,}")
    print(f"  Skipped (weight mega-cluster): {tier2_skipped_mega:,}")

    # ============================================================
    # COMBINE INTO UNIFIED ATLAS
    # ============================================================
    print("\n[3/5] Building unified atlas...")

    atlas = {}
    atlas.update(tier1)
    atlas.update(tier2)

    total_neurons = 64 * 25600  # 1,638,400
    coverage = len(atlas) / total_neurons

    # Count clusters by tier
    relp_cluster_count = len(set(cid for src, cid in atlas.values() if src == "relp"))
    weight_cluster_count = len(set(cid for src, cid in atlas.values() if src == "weight"))

    print(f"  Total atlas neurons: {len(atlas):,} ({100*coverage:.1f}% of model)")
    print(f"  RelP clusters: {relp_cluster_count:,}")
    print(f"  Weight clusters: {weight_cluster_count:,}")
    print(f"  Total clusters: {relp_cluster_count + weight_cluster_count:,}")
    print(f"  Unassigned: {total_neurons - len(atlas):,} "
          f"({100*(1-coverage):.1f}%)")

    # Cluster size distribution
    all_cluster_sizes = Counter(cid for _, (_, cid) in atlas.items())
    sizes = list(all_cluster_sizes.values())
    import numpy as np
    sizes_arr = np.array(sizes)

    print("\n  Cluster size distribution:")
    print(f"    Total: {len(sizes):,}")
    print(f"    ≥3: {sum(1 for s in sizes if s >= 3):,}")
    print(f"    ≥10: {sum(1 for s in sizes if s >= 10):,}")
    print(f"    ≥50: {sum(1 for s in sizes if s >= 50):,}")
    print(f"    Median: {np.median(sizes_arr):.0f}")
    print(f"    Mean: {sizes_arr.mean():.1f}")
    print(f"    Max: {sizes_arr.max():,}")

    # ============================================================
    # BUILD DuckDB DATABASE
    # ============================================================
    print("\n[4/5] Building DuckDB atlas...")

    db_path = "data/qwen32b_tiered_atlas.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = duckdb.connect(db_path)

    # Create neurons table
    db.execute("""
        CREATE TABLE neurons (
            layer INTEGER,
            neuron INTEGER,
            tier VARCHAR,          -- 'relp' or 'weight'
            cluster_id INTEGER,    -- unified cluster ID
            label VARCHAR,
            max_activation DOUBLE,
            PRIMARY KEY (layer, neuron)
        )
    """)

    # Create clusters table
    db.execute("""
        CREATE TABLE clusters (
            cluster_id INTEGER PRIMARY KEY,
            tier VARCHAR,
            size INTEGER,
            min_layer INTEGER,
            max_layer INTEGER,
            layer_span INTEGER
        )
    """)

    # Load labels from original DuckDB
    orig_db = duckdb.connect("data/qwen32b_neurons.duckdb", read_only=True)

    # Batch insert neurons
    print("  Inserting neurons...")
    batch = []
    for node_str, (tier, cluster_id) in atlas.items():
        parts = node_str.split("/")
        layer = int(parts[0][1:])
        neuron = int(parts[1][1:])

        row = orig_db.execute(
            "SELECT label, max_activation FROM neurons WHERE layer=? AND neuron=?",
            [layer, neuron]).fetchone()
        label = row[0] if row else None
        max_act = float(row[1]) if row and row[1] else None

        batch.append((layer, neuron, tier, cluster_id, label, max_act))

        if len(batch) >= 10000:
            db.executemany(
                "INSERT INTO neurons VALUES (?, ?, ?, ?, ?, ?)", batch)
            batch = []

    if batch:
        db.executemany("INSERT INTO neurons VALUES (?, ?, ?, ?, ?, ?)", batch)

    orig_db.close()

    # Compute cluster stats
    print("  Computing cluster stats...")
    cluster_stats = db.execute("""
        SELECT cluster_id, tier, count(*) as size,
               min(layer), max(layer), max(layer) - min(layer) + 1
        FROM neurons
        GROUP BY cluster_id, tier
    """).fetchall()

    db.executemany(
        "INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?)",
        cluster_stats)

    # Create indices
    db.execute("CREATE INDEX idx_neurons_cluster ON neurons(cluster_id)")
    db.execute("CREATE INDEX idx_neurons_tier ON neurons(tier)")
    db.execute("CREATE INDEX idx_neurons_label ON neurons(label)")

    # ============================================================
    # COPY EDGES FROM ORIGINAL DB
    # ============================================================
    print("  Copying edges from original atlas...")
    orig_db = duckdb.connect("data/qwen32b_neurons.duckdb", read_only=True)

    db.execute("""
        CREATE TABLE edges AS
        SELECT * FROM orig_db.edges
    """)
    db.execute("CREATE INDEX idx_edges_src ON edges(src_layer, src_neuron)")
    db.execute("CREATE INDEX idx_edges_tgt ON edges(tgt_layer, tgt_neuron)")

    orig_db.close()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n[5/5] Final summary...")

    stats = db.execute("""
        SELECT tier, count(*) as n_neurons, count(DISTINCT cluster_id) as n_clusters
        FROM neurons GROUP BY tier
    """).fetchall()

    total_n = db.execute("SELECT count(*) FROM neurons").fetchone()[0]
    total_c = db.execute("SELECT count(*) FROM clusters").fetchone()[0]
    total_e = db.execute("SELECT count(*) FROM edges").fetchone()[0]

    print(f"\n{'='*60}")
    print("TIERED CIRCUIT ATLAS — SUMMARY")
    print(f"{'='*60}")
    print(f"  Database: {db_path}")
    print(f"  Total neurons: {total_n:,} ({100*total_n/total_neurons:.1f}% coverage)")
    print(f"  Total clusters: {total_c:,}")
    print(f"  Total edges: {total_e:,}")
    print("\n  By tier:")
    for tier, n, c in stats:
        print(f"    {tier:>8}: {n:>10,} neurons, {c:>6,} clusters")

    size_stats = db.execute("""
        SELECT
            count(*) as n,
            sum(CASE WHEN size >= 3 THEN 1 ELSE 0 END) as ge3,
            sum(CASE WHEN size >= 10 THEN 1 ELSE 0 END) as ge10,
            sum(CASE WHEN size >= 50 THEN 1 ELSE 0 END) as ge50,
            median(size) as median_size,
            max(size) as max_size
        FROM clusters
    """).fetchone()

    print("\n  Cluster sizes:")
    print(f"    ≥3: {size_stats[1]:,}")
    print(f"    ≥10: {size_stats[2]:,}")
    print(f"    ≥50: {size_stats[3]:,}")
    print(f"    Median: {size_stats[4]}")
    print(f"    Max: {size_stats[5]:,}")

    db.close()

    elapsed = time.time() - t0
    print(f"\n  Built in {elapsed:.0f}s")
    print(f"  Use: duckdb.connect('{db_path}', read_only=True)")


if __name__ == "__main__":
    main()
