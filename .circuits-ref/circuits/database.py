"""DuckDB-backed circuit atlas: units, edges, clusters, metadata.

Supports both building a new database from aggregated data and
querying an existing one.

Usage (querying):
    from circuits.database import CircuitDatabase

    with CircuitDatabase("data/qwen32b_neurons.duckdb", read_only=True) as db:
        unit = db.get_unit(24, 5326)
        edges = db.get_edges_for_unit(24, 5326, direction="downstream")
        results = db.search_units("%enzyme%")

Usage (building):
    db = CircuitDatabase.build_from_aggregator(
        aggregator,
        labels_path=Path("data/labels.jsonl"),
        cluster_assignments=assignments,
        output_path=Path("data/atlas.duckdb"),
    )
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import duckdb

from .schemas import Edge, Unit, UnitType


class CircuitDatabase:
    """DuckDB-backed circuit atlas: units, edges, clusters, metadata.

    Supports both building a new database from aggregated data and
    querying an existing one.
    """

    def __init__(self, db_path: Path | str, read_only: bool = False):
        """Open or create a DuckDB database."""
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._conn = duckdb.connect(str(self.db_path), read_only=read_only)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> CircuitDatabase:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ---- Schema creation ------------------------------------------------

    def create_tables(self) -> None:
        """Create the neurons, edges, clusters, metadata tables if not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS neurons (
                layer INTEGER NOT NULL,
                neuron INTEGER NOT NULL,
                feature_id INTEGER,
                unit_type VARCHAR DEFAULT 'neuron',
                label VARCHAR,
                input_label VARCHAR,
                output_label VARCHAR,
                description VARCHAR,
                max_activation FLOAT,
                num_exemplars INTEGER,
                appearance_count INTEGER DEFAULT 0,
                output_norm FLOAT,
                cluster_path VARCHAR,
                top_cluster INTEGER,
                sub_module INTEGER,
                subsub_module INTEGER,
                hierarchy_depth INTEGER,
                infomap_flow FLOAT,
                PRIMARY KEY (layer, neuron)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                src_layer INTEGER NOT NULL,
                src_neuron INTEGER NOT NULL,
                tgt_layer INTEGER NOT NULL,
                tgt_neuron INTEGER NOT NULL,
                count INTEGER DEFAULT 1,
                weight_sum FLOAT,
                weight_abs_sum FLOAT,
                weight_sq_sum FLOAT,
                weight_min FLOAT,
                weight_max FLOAT,
                mean_weight FLOAT,
                mean_abs_weight FLOAT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_path VARCHAR,
                level INTEGER,
                top_cluster INTEGER,
                sub_module INTEGER,
                subsub_module INTEGER,
                size INTEGER,
                layer_min INTEGER,
                layer_max INTEGER,
                median_layer FLOAT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

    # ---- Write operations ------------------------------------------------

    def write_units(self, units: list[Unit], batch_size: int = 10000) -> int:
        """Insert units into the neurons table. Returns count inserted."""
        total = 0
        for i in range(0, len(units), batch_size):
            batch = units[i : i + batch_size]
            values = []
            for u in batch:
                values.append((
                    u.layer,
                    u.index,
                    None,  # feature_id
                    u.unit_type.value,
                    u.label or None,
                    u.input_label or None,
                    u.output_label or None,
                    (u.labels[0].description if u.labels else None),
                    u.max_activation,
                    None,  # num_exemplars
                    u.appearance_count,
                    u.output_norm,
                    u.cluster_path,
                    u.top_cluster,
                    u.sub_module,
                    u.subsub_module,
                    (len(u.cluster_path.split(".")) if u.cluster_path else None),
                    u.infomap_flow,
                ))
            self._conn.executemany(
                """INSERT INTO neurons VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                values,
            )
            total += len(batch)
        return total

    def write_edges(self, edges: list[Edge], batch_size: int = 50000) -> int:
        """Insert edges into the edges table. Returns count inserted."""
        total = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            values = []
            for e in batch:
                values.append((
                    e.src_layer,
                    e.src_index,
                    e.tgt_layer,
                    e.tgt_index,
                    e.count,
                    e.weight_sum,
                    e.weight_abs_sum,
                    e.weight_sq_sum,
                    e.weight_min,
                    e.weight_max,
                    e.mean_weight,
                    e.mean_abs_weight,
                ))
            self._conn.executemany(
                """INSERT INTO edges VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                values,
            )
            total += len(batch)
        return total

    def write_clusters(self, assignments: dict[str, Any]) -> int:
        """Write cluster assignments (from Infomap). Returns count inserted.

        ``assignments`` maps "L{layer}/{neuron}" to dicts with keys:
        path, top, sub, subsub, depth, flow.

        This method first updates matching neurons rows with cluster info,
        then builds and inserts the clusters table rows.
        """
        # Update neurons with cluster assignments
        for nkey, info in assignments.items():
            parts = nkey.split("/")
            layer = int(parts[0][1:])  # strip "L"
            neuron = int(parts[1])
            self._conn.execute(
                """UPDATE neurons SET
                    cluster_path = ?,
                    top_cluster = ?,
                    sub_module = ?,
                    subsub_module = ?,
                    hierarchy_depth = ?,
                    infomap_flow = ?
                WHERE layer = ? AND neuron = ?""",
                (
                    info.get("path"),
                    info.get("top"),
                    info.get("sub"),
                    info.get("subsub"),
                    info.get("depth"),
                    info.get("flow"),
                    layer,
                    neuron,
                ),
            )

        # Build clusters table from assignments
        from collections import defaultdict

        cluster_neurons: dict[str, list[int]] = defaultdict(list)
        for nkey, info in assignments.items():
            layer = int(nkey.split("/")[0][1:])
            path = info["path"]
            parts = path.split(".")
            for depth in range(1, len(parts) + 1):
                cpath = ".".join(parts[:depth])
                cluster_neurons[cpath].append(layer)

        values = []
        for cpath, layers in cluster_neurons.items():
            parts = cpath.split(".")
            values.append((
                cpath,
                len(parts),
                int(parts[0]),
                int(parts[1]) if len(parts) > 1 else None,
                int(parts[2]) if len(parts) > 2 else None,
                len(layers),
                min(layers),
                max(layers),
                float(statistics.median(layers)),
            ))

        self._conn.executemany(
            "INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return len(values)

    def write_metadata(self, key: str, value: str) -> None:
        """Write a key-value metadata entry (upserts)."""
        self._conn.execute(
            """INSERT INTO metadata (key, value) VALUES (?, ?)
               ON CONFLICT (key) DO UPDATE SET value = excluded.value""",
            (key, value),
        )

    def create_indexes(self) -> None:
        """Create indexes on key columns for fast lookups."""
        idx_stmts = [
            "CREATE INDEX IF NOT EXISTS idx_neurons_pk ON neurons(layer, neuron)",
            "CREATE INDEX IF NOT EXISTS idx_neurons_cluster ON neurons(top_cluster)",
            "CREATE INDEX IF NOT EXISTS idx_neurons_path ON neurons(cluster_path)",
            "CREATE INDEX IF NOT EXISTS idx_neurons_label ON neurons(label)",
            "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_layer, src_neuron)",
            "CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(tgt_layer, tgt_neuron)",
            "CREATE INDEX IF NOT EXISTS idx_clusters_top ON clusters(top_cluster)",
            "CREATE INDEX IF NOT EXISTS idx_clusters_level ON clusters(level)",
        ]
        for stmt in idx_stmts:
            self._conn.execute(stmt)

    # ---- Read operations ------------------------------------------------

    def get_unit(self, layer: int, index: int) -> Unit | None:
        """Look up a single unit by layer and index."""
        rows = self._query_to_dicts(
            "SELECT * FROM neurons WHERE layer = ? AND neuron = ?",
            (layer, index),
        )
        if not rows:
            return None
        return self._dict_to_unit(rows[0])

    def get_units_by_cluster(self, cluster_id: int) -> list[Unit]:
        """Get all units in a top-level cluster."""
        rows = self._query_to_dicts(
            "SELECT * FROM neurons WHERE top_cluster = ? ORDER BY layer, neuron",
            (cluster_id,),
        )
        return [self._dict_to_unit(r) for r in rows]

    def search_units(self, label_pattern: str, limit: int = 100) -> list[Unit]:
        """Search units by label pattern (SQL ILIKE)."""
        rows = self._query_to_dicts(
            "SELECT * FROM neurons WHERE label ILIKE ? LIMIT ?",
            (label_pattern, limit),
        )
        return [self._dict_to_unit(r) for r in rows]

    def get_edges_for_unit(
        self, layer: int, index: int, direction: str = "both"
    ) -> list[Edge]:
        """Get edges connected to a unit (upstream, downstream, or both)."""
        results = []
        if direction in ("downstream", "both"):
            rows = self._query_to_dicts(
                "SELECT * FROM edges WHERE src_layer = ? AND src_neuron = ?",
                (layer, index),
            )
            results.extend(self._dict_to_edge(r) for r in rows)
        if direction in ("upstream", "both"):
            rows = self._query_to_dicts(
                "SELECT * FROM edges WHERE tgt_layer = ? AND tgt_neuron = ?",
                (layer, index),
            )
            results.extend(self._dict_to_edge(r) for r in rows)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics (unit count, edge count, cluster count)."""
        stats = {}
        for table in ("neurons", "edges", "clusters", "metadata"):
            try:
                count = self._conn.execute(
                    f"SELECT count(*) FROM {table}"
                ).fetchone()[0]
                stats[f"{table}_count"] = count
            except duckdb.CatalogException:
                stats[f"{table}_count"] = 0

        # Get metadata entries
        try:
            rows = self._conn.execute("SELECT key, value FROM metadata").fetchall()
            stats["metadata"] = {k: v for k, v in rows}
        except duckdb.CatalogException:
            stats["metadata"] = {}

        return stats

    # ---- Build pipeline ------------------------------------------------

    @classmethod
    def build_from_aggregator(
        cls,
        aggregator,
        labels_path: Path | None = None,
        cluster_assignments: dict | None = None,
        output_path: Path = Path("data/atlas.duckdb"),
        min_edge_count: int = 3,
    ) -> CircuitDatabase:
        """Build a complete database from aggregator data, optional labels, and clusters.

        Args:
            aggregator: An InMemoryAggregator instance with .edges and .neurons dicts.
                - edges: Dict[(src_layer_str, src_neuron, tgt_layer_str, tgt_neuron) ->
                         [count, sum, abs_sum, sq_sum, min, max]]
                - neurons: Dict[(layer_str, neuron) -> count]
            labels_path: Path to a JSONL labels file. Each line has
                feature_metadata.layer, feature_metadata.neuron, labels[0].label, etc.
            cluster_assignments: Dict mapping "L{layer}/{neuron}" to cluster info dicts.
            output_path: Where to write the DuckDB file.
            min_edge_count: Minimum edge observation count to include.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file to start fresh
        if output_path.exists():
            output_path.unlink()

        db = cls(output_path)
        db.create_tables()

        # -- Load labels if provided --
        labels: dict[tuple, dict] = {}
        if labels_path is not None:
            labels = _load_labels_jsonl(labels_path)

        # -- Write neurons --
        # Iterate aggregator.neurons (keyed by (layer_str, neuron_int))
        units = []
        for (layer_str, neuron_idx), count in aggregator.neurons.items():
            try:
                layer_int = int(layer_str[1:]) if isinstance(layer_str, str) else int(layer_str)
            except (ValueError, TypeError):
                continue

            linfo = labels.get((layer_int, neuron_idx), {})
            cinfo = {}
            if cluster_assignments:
                cinfo = cluster_assignments.get(f"L{layer_int}/{neuron_idx}", {})

            unit = Unit(
                layer=layer_int,
                index=neuron_idx,
                label=linfo.get("label", ""),
                appearance_count=count,
                max_activation=linfo.get("max_activation"),
                cluster_path=cinfo.get("path"),
                top_cluster=cinfo.get("top"),
                sub_module=cinfo.get("sub"),
                subsub_module=cinfo.get("subsub"),
                infomap_flow=cinfo.get("flow"),
            )
            units.append(unit)

        n_units = db.write_units(units)

        # -- Write edges --
        valid_prefixes = {"L"}  # layer strings start with "L"
        edges = []
        for (sl, sn, tl, tn), stats in aggregator.edges.items():
            count = stats[0]
            if count < min_edge_count:
                continue
            # Filter to valid MLP layers (layer strings like "L0", "L63")
            sl_str = sl if isinstance(sl, str) else str(sl)
            tl_str = tl if isinstance(tl, str) else str(tl)
            if sl_str[0:1] not in valid_prefixes or tl_str[0:1] not in valid_prefixes:
                continue
            try:
                src_layer = int(sl_str[1:])
                tgt_layer = int(tl_str[1:])
            except (ValueError, TypeError):
                continue

            weight_sum = stats[1]
            abs_sum = stats[2]
            sq_sum = stats[3]
            w_min = stats[4]
            w_max = stats[5]

            edge = Edge(
                src_layer=src_layer,
                src_index=sn,
                tgt_layer=tgt_layer,
                tgt_index=tn,
                count=count,
                weight_sum=weight_sum,
                weight_abs_sum=abs_sum,
                weight_sq_sum=sq_sum,
                weight_min=w_min,
                weight_max=w_max,
            )
            edges.append(edge)

        n_edges = db.write_edges(edges)

        # -- Write clusters --
        n_clusters = 0
        if cluster_assignments:
            n_clusters = db.write_clusters(cluster_assignments)

        # -- Create indexes --
        db.create_indexes()

        # -- Write metadata --
        db.write_metadata("total_neurons", str(n_units))
        db.write_metadata("total_edges", str(n_edges))
        db.write_metadata("total_clusters", str(n_clusters))
        db.write_metadata("min_edge_count", str(min_edge_count))
        db.write_metadata("created", time.strftime("%Y-%m-%d %H:%M:%S"))

        return db

    # ---- Internal helpers ------------------------------------------------

    def _query_to_dicts(self, sql: str, params=None) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        Uses column names from cursor.description so results work
        regardless of schema version (old DBs without new columns).
        """
        result = self._conn.execute(sql, params or [])
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def _dict_to_unit(self, d: dict[str, Any]) -> Unit:
        """Convert a dict (column_name -> value) to a Unit."""
        ut_raw = d.get("unit_type")
        try:
            ut = UnitType(ut_raw) if ut_raw else UnitType.NEURON
        except ValueError:
            ut = UnitType.NEURON
        return Unit(
            layer=d["layer"],
            index=d["neuron"],
            unit_type=ut,
            label=d.get("label") or "",
            input_label=d.get("input_label") or "",
            output_label=d.get("output_label") or "",
            max_activation=d.get("max_activation"),
            appearance_count=d.get("appearance_count") or d.get("num_exemplars") or 0,
            output_norm=d.get("output_norm"),
            cluster_path=d.get("cluster_path"),
            top_cluster=d.get("top_cluster"),
            sub_module=d.get("sub_module"),
            subsub_module=d.get("subsub_module"),
            infomap_flow=d.get("infomap_flow"),
        )

    def _dict_to_edge(self, d: dict[str, Any]) -> Edge:
        """Convert a dict (column_name -> value) to an Edge."""
        return Edge(
            src_layer=d["src_layer"],
            src_index=d["src_neuron"],
            tgt_layer=d["tgt_layer"],
            tgt_index=d["tgt_neuron"],
            count=d.get("count") or 1,
            weight_sum=d.get("weight_sum") or 0.0,
            weight_abs_sum=d.get("weight_abs_sum") or 0.0,
            weight_sq_sum=d.get("weight_sq_sum") or 0.0,
            weight_min=d.get("weight_min") or 0.0,
            weight_max=d.get("weight_max") or 0.0,
        )

    # Backward-compat aliases for tests that use the old positional API
    def _row_to_unit(self, row) -> Unit:
        """Convert a positional row to Unit via dict conversion."""
        result = self._conn.execute("SELECT * FROM neurons LIMIT 0")
        cols = [desc[0] for desc in result.description]
        return self._dict_to_unit(dict(zip(cols, row)))

    def _row_to_edge(self, row) -> Edge:
        """Convert a positional row to Edge via dict conversion."""
        result = self._conn.execute("SELECT * FROM edges LIMIT 0")
        cols = [desc[0] for desc in result.description]
        return self._dict_to_edge(dict(zip(cols, row)))


def _load_labels_jsonl(path: Path) -> dict[tuple, dict]:
    """Load labels from a JSONL file into {(layer, neuron): info} dict."""
    labels = {}
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                meta = obj.get("feature_metadata", {})
                layer = meta.get("layer")
                neuron = meta.get("neuron")
                if layer is None or neuron is None:
                    continue

                label_list = obj.get("labels", [])
                if label_list:
                    lbl = label_list[0]
                    label_text = lbl.get("label", "")
                    parsed = lbl.get("metadata", {}).get("parsed_response", {})
                    description = parsed.get("description", "")
                else:
                    label_text = ""
                    description = ""

                labels[(layer, neuron)] = {
                    "label": label_text,
                    "description": description,
                    "max_activation": meta.get("max_activation"),
                    "num_exemplars": meta.get("num_exemplars"),
                    "feature_id": obj.get("feature_id"),
                }
            except (json.JSONDecodeError, KeyError):
                continue
    return labels
