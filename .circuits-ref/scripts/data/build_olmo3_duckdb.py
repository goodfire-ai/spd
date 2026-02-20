#!/usr/bin/env python3
"""Build DuckDB atlas for OLMo-3-7B-Instruct from enriched labels and wiring cache.

Data sources:
  - data/olmo3_enriched_labels.json (513MB, 344,551 neuron labels)
  - data/olmo3_wiring_cache/layer_{0..31}.json (3.2GB total, connectivity)

Output:
  data/olmo3_neurons.duckdb (neurons, edges, metadata tables)

Model: allenai/OLMo-3-7B-Instruct
  32 layers x 11,008 neurons = 352,256 total neurons
  Labels cover 344,551 (97.8%)
  No cluster assignments (skip clusters table)

Edge extraction uses only downstream connections from the wiring cache
to avoid double-counting. Each layer file is processed one at a time
to limit memory usage (~100MB per file).
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuits.database import CircuitDatabase
from circuits.schemas import (
    ConnectivityMethod,
    Edge,
    LabelSource,
    TokenProjection,
    Unit,
    UnitLabel,
    UnitType,
)

# ── Config ──────────────────────────────────────────────────────────────────

NUM_LAYERS = 32
NEURONS_PER_LAYER = 11_008

LABELS_PATH = Path("data/olmo3_enriched_labels.json")
WIRING_DIR = Path("data/olmo3_wiring_cache")
DEFAULT_OUTPUT = Path("data/olmo3_neurons.duckdb")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def mem_gb() -> float:
    """Current RSS in GB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        return -1.0
    return -1.0


# ── Step 1: Process wiring cache layer by layer ────────────────────────────

def process_wiring_cache(wiring_dir: Path) -> tuple[list[Unit], list[Edge]]:
    """Load wiring cache one layer at a time, extracting units and downstream edges.

    Args:
        wiring_dir: Path to directory containing layer_N.json files.

    Returns:
        (units, edges) where units have projections and edges use effective_strength.
    """
    all_units: list[Unit] = []
    all_edges: list[Edge] = []

    for layer_idx in range(NUM_LAYERS):
        layer_file = wiring_dir / f"layer_{layer_idx}.json"
        if not layer_file.exists():
            log.warning("Missing wiring file: %s", layer_file)
            continue

        t0 = time.time()
        log.info(
            "Processing layer %d/%d (%s) ...",
            layer_idx, NUM_LAYERS - 1, layer_file.name,
        )

        with open(layer_file) as f:
            layer_data = json.load(f)

        n_edges_layer = 0

        for neuron_key, entry in layer_data.items():
            neuron_idx = int(neuron_key)

            # Build output projections from the projections field
            output_projections = None
            proj = entry.get("projections")
            if proj:
                promotes = [
                    TokenProjection(
                        token=p["token"],
                        token_id=p["token_id"],
                        weight=p["weight"],
                    )
                    for p in proj.get("promotes", [])
                ]
                suppresses = [
                    TokenProjection(
                        token=p["token"],
                        token_id=p["token_id"],
                        weight=p["weight"],
                    )
                    for p in proj.get("suppresses", [])
                ]
                if promotes or suppresses:
                    output_projections = {
                        "promotes": promotes,
                        "suppresses": suppresses,
                    }

            unit = Unit(
                layer=layer_idx,
                index=neuron_idx,
                unit_type=UnitType.NEURON,
                output_projections=output_projections,
            )
            all_units.append(unit)

            # Extract downstream edges only (to avoid double-counting)
            downstream = entry.get("downstream", {})
            for conn_list_key in ("excitatory", "inhibitory"):
                for conn in downstream.get(conn_list_key, []):
                    tgt_layer = conn["layer"]
                    tgt_neuron = conn["neuron"]
                    weight = conn["effective_strength"]

                    edge = Edge(
                        src_layer=layer_idx,
                        src_index=neuron_idx,
                        tgt_layer=tgt_layer,
                        tgt_index=tgt_neuron,
                        weight=weight,
                        count=1,
                        weight_sum=weight,
                        weight_abs_sum=abs(weight),
                        weight_sq_sum=weight * weight,
                        weight_min=weight,
                        weight_max=weight,
                        method=ConnectivityMethod.WEIGHT_GRAPH,
                    )
                    all_edges.append(edge)
                    n_edges_layer += 1

        elapsed = time.time() - t0
        log.info(
            "  Layer %d: %d neurons, %d edges (%.1fs, RSS: %.1f GB)",
            layer_idx, len(layer_data), n_edges_layer, elapsed, mem_gb(),
        )

        # Free layer data before loading next layer
        del layer_data
        gc.collect()

    log.info(
        "Wiring cache complete: %d units, %d edges",
        len(all_units), len(all_edges),
    )
    return all_units, all_edges


# ── Step 2: Load enriched labels and merge ─────────────────────────────────

def load_and_merge_labels(units: list[Unit], labels_path: Path) -> list[Unit]:
    """Load enriched labels JSON and merge into existing units.

    Args:
        units: List of Unit objects from wiring cache.
        labels_path: Path to olmo3_enriched_labels.json.

    Updates each unit's label, input_label, output_label, and labels list.
    Returns units list (modified in place).
    """
    log.info("Loading enriched labels from %s ...", labels_path)
    t0 = time.time()

    with open(labels_path) as f:
        labels_data = json.load(f)

    log.info(
        "  Loaded %d labels (%.1fs, RSS: %.1f GB)",
        len(labels_data), time.time() - t0, mem_gb(),
    )

    # Build lookup from (layer, neuron) -> unit for fast merging
    unit_lookup: dict[tuple[int, int], Unit] = {}
    for unit in units:
        unit_lookup[(unit.layer, unit.index)] = unit

    n_merged = 0
    n_new = 0

    for neuron_id, entry in labels_data.items():
        # Parse neuron_id like "L31/N0"
        parts = neuron_id.split("/")
        if len(parts) != 2:
            continue
        try:
            layer = int(parts[0][1:])  # strip "L"
            neuron = int(parts[1][1:])  # strip "N"
        except (ValueError, IndexError):
            continue

        key = (layer, neuron)
        unit = unit_lookup.get(key)

        if unit is None:
            # Neuron exists in labels but not in wiring cache -- create unit
            unit = Unit(layer=layer, index=neuron, unit_type=UnitType.NEURON)
            units.append(unit)
            unit_lookup[key] = unit
            n_new += 1

        # Merge label fields
        short_label = entry.get("short_label", "")
        full_label = entry.get("label", "")
        unit.label = short_label or full_label
        unit.input_label = entry.get("input_label", "")
        unit.output_label = full_label  # full descriptive label as output label

        # Build UnitLabel entries for provenance tracking
        unit.labels = []

        # Output label from autointerp enrichment
        if full_label:
            unit.labels.append(UnitLabel(
                text=full_label,
                source=LabelSource.AUTOINTERP,
                confidence=entry.get("input_confidence", "llm-auto"),
                description=entry.get("output_function", ""),
            ))

        # Input label from enrichment
        input_label = entry.get("input_label", "")
        if input_label:
            unit.labels.append(UnitLabel(
                text=input_label,
                source=LabelSource.PROGRESSIVE_INPUT,
                confidence=entry.get("input_confidence", "llm-auto"),
                description=entry.get("input_function", ""),
            ))

        # Original autointerp label (raw, before enrichment)
        autointerp_label = entry.get("autointerp_label", "")
        if autointerp_label:
            unit.labels.append(UnitLabel(
                text=autointerp_label,
                source=LabelSource.AUTOINTERP,
                confidence="llm-auto",
                description="Original autointerp label before enrichment",
            ))

        n_merged += 1

    log.info("  Merged %d labels, created %d new units", n_merged, n_new)
    log.info("  Total units: %d", len(units))
    return units


# ── Step 3: Write to DuckDB ───────────────────────────────────────────────

def write_database(
    units: list[Unit],
    edges: list[Edge],
    output_path: Path,
) -> None:
    """Write units, edges, and metadata to DuckDB."""

    # Remove existing file to start fresh
    if output_path.exists():
        log.info("Removing existing database: %s", output_path)
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Creating database: %s", output_path)
    db = CircuitDatabase(output_path)
    db.create_tables()

    # Write units
    log.info("Writing %d units ...", len(units))
    t0 = time.time()
    n_units = db.write_units(units)
    log.info("  Wrote %d units (%.1fs)", n_units, time.time() - t0)

    # Write edges
    log.info("Writing %d edges ...", len(edges))
    t0 = time.time()
    n_edges = db.write_edges(edges)
    log.info("  Wrote %d edges (%.1fs)", n_edges, time.time() - t0)

    # Write metadata
    log.info("Writing metadata ...")
    db.write_metadata("model", "allenai/OLMo-3-7B-Instruct")
    db.write_metadata("num_layers", str(NUM_LAYERS))
    db.write_metadata("neurons_per_layer", str(NEURONS_PER_LAYER))
    db.write_metadata("total_neurons", str(n_units))
    db.write_metadata("total_edges", str(n_edges))
    db.write_metadata("edge_method", "weight_graph")
    db.write_metadata("source_labels", "olmo3_enriched_labels.json")
    db.write_metadata("source_wiring", "olmo3_wiring_cache/")
    db.write_metadata("label_source", "OLMo-3 enriched autointerp labels")
    db.write_metadata("created", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Create indexes
    log.info("Creating indexes ...")
    t0 = time.time()
    db.create_indexes()
    log.info("  Indexes created (%.1fs)", time.time() - t0)

    # ── Summary ──
    stats = db.get_stats()
    log.info("")
    log.info("════════ Database Summary ════════")
    for table in ("neurons", "edges", "clusters", "metadata"):
        count = stats.get(f"{table}_count", 0)
        log.info("  %-10s %s rows", table + ":", f"{count:,}")

    # Label coverage
    result = db._conn.execute(
        "SELECT count(*) FROM neurons WHERE label IS NOT NULL AND label != ''"
    ).fetchone()
    n_labeled = result[0] if result else 0
    pct = 100 * n_labeled / n_units if n_units > 0 else 0
    log.info("  Labels:    %s / %s (%.1f%%)", f"{n_labeled:,}", f"{n_units:,}", pct)

    # Edge stats
    result = db._conn.execute(
        "SELECT min(weight_min), max(weight_max), avg(mean_weight) FROM edges"
    ).fetchone()
    if result and result[0] is not None:
        log.info(
            "  Edge weights: min=%.4f, max=%.4f, avg_mean=%.4f",
            result[0], result[1], result[2],
        )

    # Layer distribution
    result = db._conn.execute(
        "SELECT layer, count(*) FROM neurons GROUP BY layer ORDER BY layer"
    ).fetchall()
    layers_with_neurons = len(result)
    log.info("  Layers with neurons: %d / %d", layers_with_neurons, NUM_LAYERS)

    # Sample neurons
    log.info("")
    log.info("Sample neurons:")
    sample = db._conn.execute(
        """SELECT layer, neuron, label, input_label, output_label
           FROM neurons
           WHERE label IS NOT NULL AND label != ''
           ORDER BY random()
           LIMIT 10"""
    ).fetchall()
    for row in sample:
        log.info(
            "  L%d/N%d: label=%r  input=%r  output=%r",
            row[0], row[1],
            (row[2] or "")[:60],
            (row[3] or "")[:40],
            (row[4] or "")[:40],
        )

    db_size = output_path.stat().st_size / (1024 * 1024)
    log.info("")
    log.info("Output: %s (%.1f MB)", output_path, db_size)

    db.close()


# ── Step 4: Print summary ─────────────────────────────────────────────────

def print_summary(units: list[Unit], edges: list[Edge]) -> None:
    """Print data extraction summary before writing."""
    n_labeled = sum(1 for u in units if u.label)
    n_with_projections = sum(1 for u in units if u.output_projections)
    n_with_input_label = sum(1 for u in units if u.input_label)

    log.info("")
    log.info("════════ Extraction Summary ════════")
    log.info("  Total units:          %s", f"{len(units):,}")
    log.info("  With labels:          %s", f"{n_labeled:,}")
    log.info("  With input labels:    %s", f"{n_with_input_label:,}")
    log.info("  With projections:     %s", f"{n_with_projections:,}")
    log.info("  Total edges:          %s", f"{len(edges):,}")
    log.info("  RSS:                  %.1f GB", mem_gb())


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build DuckDB atlas for OLMo-3-7B-Instruct",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output DuckDB path (default: %(default)s)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH,
        help="Enriched labels JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--wiring-dir",
        type=Path,
        default=WIRING_DIR,
        help="Wiring cache directory (default: %(default)s)",
    )
    args = parser.parse_args()

    labels_path = args.labels
    wiring_dir = args.wiring_dir

    log.info("═══════════════════════════════════════════════════")
    log.info(" OLMo-3-7B-Instruct DuckDB Builder")
    log.info("═══════════════════════════════════════════════════")
    log.info("  Labels:  %s", labels_path)
    log.info("  Wiring:  %s", wiring_dir)
    log.info("  Output:  %s", args.output)
    log.info("  RSS:     %.1f GB", mem_gb())
    log.info("")

    t_start = time.time()

    # Step 1: Process wiring cache (layer by layer)
    units, edges = process_wiring_cache(wiring_dir)

    # Step 2: Load enriched labels and merge
    units = load_and_merge_labels(units, labels_path)

    # Print extraction summary
    print_summary(units, edges)

    # Step 3: Write to DuckDB
    write_database(units, edges, args.output)

    elapsed = time.time() - t_start
    log.info("")
    log.info("═══════════════════════════════════════════════════")
    log.info(" DONE in %.1f min (RSS: %.1f GB)", elapsed / 60, mem_gb())
    log.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
