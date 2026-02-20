#!/usr/bin/env python3
"""Build DuckDB atlas for Llama-3.1-8B-Instruct from enriched JSON data.

Data sources:
  1. data/fineweb_50k_edge_stats_enriched.json (1.19GB) — 173,923 neuron profiles
     with edges, transluce labels, output/input projections
  2. data/neuron_labels_combined.json (160MB) — 48,187 neurons with GPT labels
     (function_label, function_description, input_label, etc.)
  3. data/neuron_clusters/llama8b_infomap_full_edges.json (2.5MB) — 172,346 cluster
     assignments (single-level Infomap, top_cluster only)

Output:
  data/llama8b_neurons.duckdb — DuckDB atlas with neurons, edges, clusters, metadata

Tables follow the CircuitDatabase schema (circuits.database).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuits.database import CircuitDatabase
from circuits.schemas import Edge, LabelSource, Unit, UnitLabel, UnitType

# ── Config ───────────────────────────────────────────────────────────────────

ENRICHED_PATH = "data/fineweb_50k_edge_stats_enriched.json"
LABELS_PATH = "data/neuron_labels_combined.json"
CLUSTERS_PATH = "data/neuron_clusters/llama8b_infomap_full_edges.json"

NUM_LAYERS = 32
NEURONS_PER_LAYER = 14336  # Llama-3.1-8B MLP intermediate size


def ts():
    return time.strftime("%H:%M:%S")


def pf(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def parse_neuron_id(neuron_id: str):
    """Parse 'L15/N7890' into (layer=15, neuron=7890)."""
    parts = neuron_id.split("/")
    layer = int(parts[0][1:])  # strip 'L'
    neuron = int(parts[1][1:])  # strip 'N'
    return layer, neuron


def parse_underscore_id(uid: str):
    """Parse '24_5326' into (layer=24, neuron=5326)."""
    parts = uid.split("_")
    return int(parts[0]), int(parts[1])


# ── Phase 1: Load data ──────────────────────────────────────────────────────

def load_enriched(path: str) -> dict:
    """Load the enriched edge stats JSON (1.19GB)."""
    pf(f"[{ts()}] Phase 1a: Loading enriched profiles from {path}...")
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    elapsed = time.time() - t0
    n_profiles = len(data.get("profiles", []))
    pf(f"[{ts()}]   Loaded {n_profiles:,} profiles in {elapsed:.1f}s")
    pf(f"[{ts()}]   Metadata: {json.dumps(data.get('metadata', {}))}")
    return data


def load_labels(path: str) -> dict:
    """Load the combined labels JSON (160MB)."""
    pf(f"[{ts()}] Phase 1b: Loading GPT labels from {path}...")
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    elapsed = time.time() - t0
    neurons = data.get("neurons", {})
    pf(f"[{ts()}]   Loaded {len(neurons):,} labeled neurons in {elapsed:.1f}s")
    return neurons


def load_clusters(path: str) -> dict:
    """Load Infomap cluster assignments (2.5MB)."""
    pf(f"[{ts()}] Phase 1c: Loading cluster assignments from {path}...")
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    elapsed = time.time() - t0
    assignments = data.get("neuron_assignments", {})
    pf(f"[{ts()}]   Loaded {len(assignments):,} assignments in {elapsed:.1f}s")
    pf(f"[{ts()}]   Method: {data.get('method', 'unknown')}")
    pf(f"[{ts()}]   Total clusters: {data.get('stats', {}).get('total_clusters', '?')}")
    return assignments


# ── Phase 2: Build Units ─────────────────────────────────────────────────────

def build_units(profiles: list, labels: dict, clusters: dict) -> list:
    """Build Unit objects from profiles, labels, and cluster assignments.

    For each of the 173,923 profiles in the enriched file:
    - Use function_label from labels file if available, else transluce_label_positive
    - Set input_label / output_label from labels file
    - Add UnitLabel entries from multiple sources
    - Set cluster assignment (top_cluster only, single-level for Llama)
    """
    pf(f"[{ts()}] Phase 2: Building Unit objects from {len(profiles):,} profiles...")
    t0 = time.time()

    units = []
    n_with_label = 0
    n_with_gpt_label = 0
    n_with_cluster = 0

    for i, profile in enumerate(profiles):
        if i > 0 and i % 50000 == 0:
            pf(f"[{ts()}]   {i:,}/{len(profiles):,} processed...")

        layer = profile["layer"]
        neuron = profile["neuron"]
        neuron_id = profile.get("neuron_id", f"L{layer}/N{neuron}")

        # Look up GPT labels
        linfo = labels.get(neuron_id, {})

        # Best available label: prefer GPT function_label, fall back to transluce
        gpt_label = linfo.get("function_label", "")
        transluce_label = profile.get("transluce_label_positive", "")
        best_label = gpt_label if gpt_label else transluce_label

        if best_label:
            n_with_label += 1
        if gpt_label:
            n_with_gpt_label += 1

        # Input and output labels from GPT labels file
        input_label = linfo.get("input_label", "")
        output_label = ""  # Not directly available; function_label covers output

        # Build provenance labels list
        unit_labels = []
        if transluce_label:
            unit_labels.append(UnitLabel(
                text=transluce_label,
                source=LabelSource.NEURONDB,
                confidence="medium",
                description=profile.get("transluce_label_negative", ""),
            ))
        if gpt_label:
            unit_labels.append(UnitLabel(
                text=gpt_label,
                source=LabelSource.AUTOINTERP,
                confidence=linfo.get("confidence", "unknown"),
                description=linfo.get("function_description", ""),
            ))
        neurondb_label = linfo.get("neurondb_label", "")
        if neurondb_label and neurondb_label != transluce_label:
            unit_labels.append(UnitLabel(
                text=neurondb_label,
                source=LabelSource.NEURONDB,
                confidence="medium",
                description="",
            ))

        # Cluster assignment (single-level for Llama)
        cluster_id = clusters.get(neuron_id)
        cluster_path = None
        top_cluster = None
        if cluster_id is not None:
            cluster_path = str(cluster_id)
            top_cluster = cluster_id
            n_with_cluster += 1

        # Build Unit
        unit = Unit(
            layer=layer,
            index=neuron,
            unit_type=UnitType.NEURON,
            label=best_label,
            input_label=input_label,
            output_label=output_label,
            labels=unit_labels,
            max_activation=linfo.get("max_activation") if linfo else None,
            appearance_count=profile.get("appearance_count", 0),
            output_norm=linfo.get("output_norm"),
            cluster_path=cluster_path,
            top_cluster=top_cluster,
            sub_module=None,  # single-level clustering for Llama
            subsub_module=None,
        )
        units.append(unit)

    elapsed = time.time() - t0
    pf(f"[{ts()}]   Built {len(units):,} units in {elapsed:.1f}s")
    pf(f"[{ts()}]   With any label: {n_with_label:,}")
    pf(f"[{ts()}]   With GPT label: {n_with_gpt_label:,}")
    pf(f"[{ts()}]   With cluster: {n_with_cluster:,}")

    return units


# ── Phase 3: Build Edges ─────────────────────────────────────────────────────

def build_edges(profiles: list) -> list:
    """Build Edge objects from top_downstream_targets in each profile.

    Only uses downstream edges to avoid double-counting (an edge A->B may
    appear in A's downstream AND B's upstream lists).

    Format: top_downstream_targets = [
        {"target": "24_5326", "count": 122176, "frequency": 3.10, "avg_weight": -0.179},
        ...
    ]
    """
    pf(f"[{ts()}] Phase 3: Building Edge objects from downstream targets...")
    t0 = time.time()

    edges = []
    n_skipped = 0

    for i, profile in enumerate(profiles):
        if i > 0 and i % 50000 == 0:
            pf(f"[{ts()}]   {i:,}/{len(profiles):,} profiles, {len(edges):,} edges so far...")

        src_layer = profile["layer"]
        src_neuron = profile["neuron"]

        for target_info in profile.get("top_downstream_targets", []):
            target_str = target_info.get("target", "")
            try:
                tgt_layer, tgt_neuron = parse_underscore_id(target_str)
            except (ValueError, IndexError):
                n_skipped += 1
                continue

            count = target_info.get("count", 1)
            avg_weight = target_info.get("avg_weight", 0.0)

            # We only have avg_weight and count from the top-10 lists.
            # Reconstruct what we can: weight_sum = avg_weight * count
            weight_sum = avg_weight * count
            abs_weight = abs(avg_weight)
            weight_abs_sum = abs_weight * count

            edge = Edge(
                src_layer=src_layer,
                src_index=src_neuron,
                tgt_layer=tgt_layer,
                tgt_index=tgt_neuron,
                count=count,
                weight_sum=weight_sum,
                weight_abs_sum=weight_abs_sum,
                weight_sq_sum=avg_weight * avg_weight * count,  # approximate
                weight_min=avg_weight,  # only have avg, use as proxy
                weight_max=avg_weight,
            )
            edges.append(edge)

    elapsed = time.time() - t0
    pf(f"[{ts()}]   Built {len(edges):,} edges in {elapsed:.1f}s")
    if n_skipped:
        pf(f"[{ts()}]   Skipped {n_skipped:,} unparseable targets")

    return edges


# ── Phase 4: Build cluster assignments dict for write_clusters ───────────────

def build_cluster_assignments(clusters: dict) -> dict:
    """Convert {neuron_id: cluster_int} to the format write_clusters expects.

    write_clusters expects:
        {"L0/N491": {"path": "42", "top": 42, "sub": None, "subsub": None,
                     "depth": 1, "flow": None}}
    """
    pf(f"[{ts()}] Phase 4: Building cluster assignments dict...")
    t0 = time.time()

    assignments = {}
    for neuron_id, cluster_id in clusters.items():
        # neuron_id is "L0/N0" but write_clusters expects "L0/0" format
        # Actually, looking at write_clusters: it splits on "/" and does parts[0][1:] for layer
        # and parts[1] for neuron. So for "L0/N0" it would get layer=0 but neuron="N0" which fails.
        # Need to convert to "L0/0" format.
        layer, neuron = parse_neuron_id(neuron_id)
        key = f"L{layer}/{neuron}"

        assignments[key] = {
            "path": str(cluster_id),
            "top": cluster_id,
            "sub": None,
            "subsub": None,
            "depth": 1,
            "flow": None,
        }

    elapsed = time.time() - t0
    pf(f"[{ts()}]   Built {len(assignments):,} cluster assignments in {elapsed:.1f}s")

    # Stats
    cluster_sizes = defaultdict(int)
    for info in assignments.values():
        cluster_sizes[info["top"]] += 1
    sizes = list(cluster_sizes.values())
    pf(f"[{ts()}]   Unique clusters: {len(cluster_sizes):,}")
    if sizes:
        pf(f"[{ts()}]   Max cluster size: {max(sizes):,}")
        pf(f"[{ts()}]   Clusters >100: {sum(1 for s in sizes if s > 100)}")
        pf(f"[{ts()}]   Clusters >10: {sum(1 for s in sizes if s > 10)}")

    return assignments


# ── Phase 5: Write to DuckDB ────────────────────────────────────────────────

def write_database(units, edges, cluster_assignments, output_path, enriched_metadata):
    """Write everything to DuckDB using CircuitDatabase."""
    pf(f"[{ts()}] Phase 5: Writing DuckDB to {output_path}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.exists():
        output_path.unlink()
        pf(f"[{ts()}]   Removed existing {output_path}")

    db = CircuitDatabase(output_path)
    db.create_tables()

    # Write units
    pf(f"[{ts()}]   Writing {len(units):,} units...")
    t0 = time.time()
    n_units = db.write_units(units)
    pf(f"[{ts()}]   Wrote {n_units:,} units in {time.time() - t0:.1f}s")

    # Write edges
    pf(f"[{ts()}]   Writing {len(edges):,} edges...")
    t0 = time.time()
    n_edges = db.write_edges(edges)
    pf(f"[{ts()}]   Wrote {n_edges:,} edges in {time.time() - t0:.1f}s")

    # Write clusters
    pf(f"[{ts()}]   Writing cluster assignments...")
    t0 = time.time()
    n_clusters = db.write_clusters(cluster_assignments)
    pf(f"[{ts()}]   Wrote {n_clusters:,} cluster rows in {time.time() - t0:.1f}s")

    # Create indexes
    pf(f"[{ts()}]   Creating indexes...")
    t0 = time.time()
    db.create_indexes()
    pf(f"[{ts()}]   Indexes created in {time.time() - t0:.1f}s")

    # Write metadata
    total_graphs = enriched_metadata.get("total_graphs", 39358)
    db.write_metadata("model", "meta-llama/Llama-3.1-8B-Instruct")
    db.write_metadata("source", "fineweb_50k_edge_stats_enriched.json")
    db.write_metadata("labels_source", "neuron_labels_combined.json (GPT two-pass + v6 fallback)")
    db.write_metadata("clusters_source", "llama8b_infomap_full_edges.json")
    db.write_metadata("total_graphs", str(total_graphs))
    db.write_metadata("total_neurons", str(n_units))
    db.write_metadata("total_edges", str(n_edges))
    db.write_metadata("total_clusters", str(n_clusters))
    db.write_metadata("edge_source", "top-10 downstream per neuron (not full edge table)")
    db.write_metadata("num_layers", str(NUM_LAYERS))
    db.write_metadata("neurons_per_layer", str(NEURONS_PER_LAYER))
    db.write_metadata("created", time.strftime("%Y-%m-%d %H:%M:%S"))

    db.close()

    db_size = os.path.getsize(output_path) / (1024 * 1024)
    pf(f"[{ts()}]   Output: {output_path} ({db_size:.1f} MB)")

    return n_units, n_edges, n_clusters


# ── Phase 6: Summary ────────────────────────────────────────────────────────

def print_summary(output_path):
    """Print summary stats from the built database."""
    pf(f"\n[{ts()}] ══════ DuckDB Summary ══════")

    db = CircuitDatabase(output_path, read_only=True)
    stats = db.get_stats()

    for table in ("neurons", "edges", "clusters", "metadata"):
        count = stats.get(f"{table}_count", 0)
        pf(f"[{ts()}]   {table}: {count:,} rows")

    # Sample queries
    pf(f"\n[{ts()}]   Top 10 clusters by size:")
    rows = db._conn.execute("""
        SELECT top_cluster, count(*) as n,
               min(layer) as l_min, max(layer) as l_max,
               list(label ORDER BY appearance_count DESC NULLS LAST)[:3] as top_labels
        FROM neurons
        WHERE top_cluster IS NOT NULL
        GROUP BY top_cluster
        ORDER BY n DESC
        LIMIT 10
    """).fetchall()
    for row in rows:
        labels_str = ", ".join(str(l) for l in (row[4] or [])[:3])
        pf(f"[{ts()}]     Cluster {row[0]:>5}: {row[1]:>7,} neurons  L{row[2]}-L{row[3]}")
        pf(f"[{ts()}]       Labels: {labels_str[:120]}")

    # Sample labeled neurons
    pf(f"\n[{ts()}]   Sample labeled neurons:")
    sample = db._conn.execute("""
        SELECT layer, neuron, label, top_cluster, appearance_count
        FROM neurons
        WHERE label IS NOT NULL AND label != ''
        ORDER BY appearance_count DESC
        LIMIT 10
    """).fetchall()
    for layer, neuron, label, cluster, count in sample:
        pf(f"[{ts()}]     L{layer}/N{neuron}  cluster={cluster}  count={count:,}  {label[:80]}")

    # Label coverage
    total = db._conn.execute("SELECT count(*) FROM neurons").fetchone()[0]
    labeled = db._conn.execute(
        "SELECT count(*) FROM neurons WHERE label IS NOT NULL AND label != ''"
    ).fetchone()[0]
    clustered = db._conn.execute(
        "SELECT count(*) FROM neurons WHERE top_cluster IS NOT NULL"
    ).fetchone()[0]
    pf(f"\n[{ts()}]   Label coverage: {labeled:,}/{total:,} ({100*labeled/total:.1f}%)")
    pf(f"[{ts()}]   Cluster coverage: {clustered:,}/{total:,} ({100*clustered/total:.1f}%)")

    db.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build DuckDB atlas for Llama-3.1-8B-Instruct from enriched JSON data."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/llama8b_neurons.duckdb",
        help="Output DuckDB path (default: data/llama8b_neurons.duckdb)",
    )
    parser.add_argument(
        "--enriched",
        default=ENRICHED_PATH,
        help=f"Path to enriched edge stats JSON (default: {ENRICHED_PATH})",
    )
    parser.add_argument(
        "--labels",
        default=LABELS_PATH,
        help=f"Path to combined labels JSON (default: {LABELS_PATH})",
    )
    parser.add_argument(
        "--clusters",
        default=CLUSTERS_PATH,
        help=f"Path to Infomap cluster assignments JSON (default: {CLUSTERS_PATH})",
    )
    args = parser.parse_args()

    pf(f"[{ts()}] ═══════════════════════════════════════════════════")
    pf(f"[{ts()}]  Llama 8B Neuron DuckDB Builder")
    pf(f"[{ts()}] ═══════════════════════════════════════════════════")
    t_start = time.time()

    # Phase 1: Load all data
    enriched_data = load_enriched(args.enriched)
    profiles = enriched_data["profiles"]
    enriched_metadata = enriched_data.get("metadata", {})

    labels = load_labels(args.labels)
    clusters = load_clusters(args.clusters)

    # Free the raw enriched container (keep profiles reference)
    del enriched_data

    # Phase 2: Build units
    units = build_units(profiles, labels, clusters)

    # Phase 3: Build edges (from downstream targets only)
    edges = build_edges(profiles)

    # Free profiles
    del profiles

    # Phase 4: Build cluster assignments dict
    cluster_assignments = build_cluster_assignments(clusters)
    del clusters

    # Phase 5: Write to DuckDB
    n_units, n_edges, n_clusters = write_database(
        units, edges, cluster_assignments, args.output, enriched_metadata
    )
    del units, edges, cluster_assignments, labels

    # Phase 6: Summary
    print_summary(args.output)

    total_time = time.time() - t_start
    pf(f"\n[{ts()}] ═══════════════════════════════════════════════════")
    pf(f"[{ts()}]  DONE in {total_time:.1f}s ({total_time/60:.1f} min)")
    pf(f"[{ts()}] ═══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
