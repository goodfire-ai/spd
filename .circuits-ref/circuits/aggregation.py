"""In-memory edge aggregation from RelP attribution graphs.

O(1) dict-based aggregation with checkpoint/resume and DuckDB/Parquet export.
Replaces SQLite UPSERT bottleneck: ~50,000 graphs/hr vs ~1,000/hr at scale.

Usage:
    from circuits.aggregation import InMemoryAggregator

    agg = InMemoryAggregator(Path("graphs/qwen3_32b_800k/"))
    agg.process_directory(Path("graphs/qwen3_32b_800k/"))
    edges = agg.get_edges(min_count=5)
    agg.export_to_duckdb(Path("data/edge_stats.duckdb"))
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

from .schemas import ConnectivityMethod, Edge, Unit

# Try msgpack for backward compat with v1 checkpoints
try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

logger = logging.getLogger(__name__)

# Internal type aliases
_EdgeKey = tuple[int, int, int, int]  # (src_layer, src_neuron, tgt_layer, tgt_neuron)
_EdgeStats = list[float]  # [count, sum, abs_sum, sq_sum, min, max]
_NeuronKey = tuple[int, int]  # (layer, neuron)


def _parse_node_id(node_id: str) -> tuple[int, int] | None:
    """Parse a legacy node ID to (layer_int, feature_int).

    Legacy format: "{layer}_{feature}_{position}" e.g. "24_5326_7".
    Returns None for embedding (E_*) or logit (L_*) nodes.
    """
    parts = node_id.split("_")
    if len(parts) < 2:
        return None
    try:
        layer = int(parts[0])
        feature = int(parts[1])
        return (layer, feature)
    except ValueError:
        return None


def _legacy_layer_to_int(layer_str: str) -> int | None:
    """Convert legacy string layer key to integer.

    Handles: "L24" -> 24, "24" -> 24, "E" -> None, "L" -> None.
    """
    if layer_str.startswith("L") and len(layer_str) > 1:
        try:
            return int(layer_str[1:])
        except ValueError:
            return None
    try:
        return int(layer_str)
    except ValueError:
        return None


class InMemoryAggregator:
    """O(1) edge aggregation from many RelP graphs with checkpoint/resume.

    Stores running statistics in plain dicts for maximum performance.
    Only converts to schema objects (Edge, Unit) in output methods.

    Data structures:
        edges: dict[(src_layer, src_neuron, tgt_layer, tgt_neuron) ->
                     [count, sum, abs_sum, sq_sum, min, max]]
        neurons: dict[(layer, neuron) -> count]
        neuron_graphs: dict[(layer, neuron) -> list[tuple[graph_file, corpus_index]]]
        processed_files: set[str]
    """

    def __init__(
        self,
        graph_dir: Path,
        checkpoint_dir: Path | None = None,
    ):
        self.graph_dir = Path(graph_dir)
        self.checkpoint_dir = checkpoint_dir or (self.graph_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Edge statistics: O(1) lookup and update
        self.edges: dict[_EdgeKey, _EdgeStats] = {}

        # Neuron appearance counts
        self.neurons: dict[_NeuronKey, int] = {}

        # Inverted index: neuron -> graphs it appeared in
        self.neuron_graphs: dict[_NeuronKey, list[tuple[str, int]]] = {}

        # Processed file tracking for resume
        self.processed_files: set[str] = set()

        # Counters
        self.graphs_processed: int = 0
        self.total_edge_observations: int = 0
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def _update_edge(self, key: _EdgeKey, weight: float) -> None:
        """Update edge running statistics — O(1)."""
        stats = self.edges.get(key)
        if stats is not None:
            stats[0] += 1
            stats[1] += weight
            stats[2] += abs(weight)
            stats[3] += weight * weight
            if weight < stats[4]:
                stats[4] = weight
            if weight > stats[5]:
                stats[5] = weight
        else:
            self.edges[key] = [1, weight, abs(weight), weight * weight, weight, weight]
        self.total_edge_observations += 1

    def process_graph(self, graph_path: Path) -> None:
        """Process a single RelP graph JSON file.

        Parses nodes and links, updates running edge/neuron statistics.
        Skips embedding (E_*) and logit (L_*) links.
        """
        graph_path = Path(graph_path)
        fname = graph_path.name

        with open(graph_path) as f:
            graph = json.load(f)

        corpus_index = graph.get("metadata", {}).get("corpus_index", -1)

        # Process edges (skip embedding/logit links)
        for link in graph.get("links", []):
            try:
                src = _parse_node_id(link["source"])
                tgt = _parse_node_id(link["target"])
                if src is None or tgt is None:
                    continue
                weight = link["weight"]
                key = (src[0], src[1], tgt[0], tgt[1])
                self._update_edge(key, weight)
            except (KeyError, TypeError):
                continue

        # Process neurons (deduplicate within graph)
        neurons_seen: set[_NeuronKey] = set()
        for node in graph.get("nodes", []):
            parsed = _parse_node_id(node.get("node_id", ""))
            if parsed is not None:
                neurons_seen.add(parsed)

        for layer, neuron in neurons_seen:
            nkey = (layer, neuron)
            self.neurons[nkey] = self.neurons.get(nkey, 0) + 1
            if nkey not in self.neuron_graphs:
                self.neuron_graphs[nkey] = []
            self.neuron_graphs[nkey].append((fname, corpus_index))

        self.processed_files.add(fname)
        self.graphs_processed += 1

    def process_directory(
        self,
        graph_dir: Path,
        batch_size: int = 100,
        checkpoint_interval: int = 500,
        max_graphs: int = 0,
    ) -> None:
        """Process all graph files in directory with periodic checkpointing.

        Args:
            graph_dir: Directory containing graph_*.json files.
            batch_size: Not used for batching; kept for API compat. All
                new files are processed in one pass.
            checkpoint_interval: Save checkpoint every N graphs.
            max_graphs: Stop after this many graphs (0 = unlimited).
        """
        graph_dir = Path(graph_dir)
        all_files = sorted(f.name for f in graph_dir.glob("graph_*.json"))
        new_files = [f for f in all_files if f not in self.processed_files]

        if not new_files:
            logger.info("No new files to process")
            return

        if max_graphs > 0:
            new_files = new_files[:max_graphs]

        logger.info(
            f"Processing {len(new_files)} new files "
            f"({len(self.edges):,} edges already in memory)"
        )

        processed_count = 0
        errors = 0
        batch_start = time.time()
        last_checkpoint_at = self.graphs_processed

        for fname in new_files:
            fpath = graph_dir / fname
            try:
                self.process_graph(fpath)
                processed_count += 1
            except (OSError, json.JSONDecodeError) as ex:
                logger.debug(f"Skipping {fname}: {ex}")
                errors += 1
                continue

            # Periodic checkpoint
            if (
                self.graphs_processed - last_checkpoint_at >= checkpoint_interval
                and self.graphs_processed > 0
            ):
                self.checkpoint(self.checkpoint_dir / f"checkpoint_{self.graphs_processed}.dat")
                last_checkpoint_at = self.graphs_processed

                elapsed = time.time() - batch_start
                rate = processed_count / elapsed * 3600 if elapsed > 0 else 0
                logger.info(
                    f"Progress: {processed_count}/{len(new_files)} | "
                    f"{len(self.edges):,} edges | {rate:,.0f}/hr"
                )

        # Final checkpoint
        if processed_count > 0:
            self.checkpoint(self.checkpoint_dir / f"checkpoint_{self.graphs_processed}.dat")

        elapsed = time.time() - batch_start
        rate = processed_count / elapsed * 3600 if elapsed > 0 else 0
        logger.info(
            f"Completed: {processed_count} files in {elapsed:.1f}s "
            f"({errors} skipped, {rate:,.0f}/hr)"
        )
        logger.info(
            f"Total: {len(self.edges):,} edges, {len(self.neurons):,} neurons"
        )

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def checkpoint(self, path: Path) -> None:
        """Save state to pickle checkpoint (v2 format).

        Uses atomic write (temp file + rename) for crash safety.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")

        logger.info(f"Saving checkpoint to {path}...")
        start = time.time()

        with open(temp_path, "wb") as f:
            metadata = {
                "graphs_processed": self.graphs_processed,
                "total_edge_observations": self.total_edge_observations,
                "version": 2,
            }
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"  Writing {len(self.edges):,} edges...")
            pickle.dump(self.edges, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"  Writing {len(self.neurons):,} neurons...")
            pickle.dump(self.neurons, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"  Writing {len(self.processed_files):,} processed files...")
            pickle.dump(self.processed_files, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"  Writing neuron_graphs index ({len(self.neuron_graphs):,} neurons)...")
            pickle.dump(dict(self.neuron_graphs), f, protocol=pickle.HIGHEST_PROTOCOL)

        temp_path.rename(path)

        elapsed = time.time() - start
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info(f"Checkpoint saved: {size_mb:.1f}MB in {elapsed:.1f}s")

        # Update "latest" symlink
        latest_link = path.parent / "latest.dat"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(path.name)

    def resume(self, path: Path) -> None:
        """Load state from checkpoint (handles v0, v1 msgpack, v2 pickle).

        Raises FileNotFoundError if the checkpoint does not exist.
        If path is a directory, looks for ``latest.dat`` symlink or
        the most recent ``checkpoint_*.dat`` file.
        """
        path = Path(path)

        # If given a directory, find the best checkpoint
        if path.is_dir():
            latest = path / "latest.dat"
            if latest.exists():
                path = latest.resolve()
            else:
                checkpoints = sorted(
                    path.glob("checkpoint_*.dat"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoint found in {path}")
                path = checkpoints[0]

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint from {path}...")
        start = time.time()

        with open(path, "rb") as f:
            first_obj = pickle.load(f)

            if isinstance(first_obj, dict) and first_obj.get("version") == 2:
                self._load_v2(first_obj, f)
            elif _HAS_MSGPACK and isinstance(first_obj, dict) and "edges" in first_obj:
                self._load_v1_msgpack(first_obj)
            elif isinstance(first_obj, dict):
                self._load_v0_pickle(first_obj)
            else:
                raise ValueError(f"Unknown checkpoint format (first object type: {type(first_obj)})")

        elapsed = time.time() - start
        logger.info(
            f"Checkpoint loaded: {len(self.edges):,} edges, "
            f"{len(self.processed_files):,} files in {elapsed:.1f}s"
        )

    def _load_v2(self, metadata: dict, f) -> None:
        """Load v2 pickle format (sequential objects)."""
        self.graphs_processed = metadata["graphs_processed"]
        self.total_edge_observations = metadata["total_edge_observations"]

        logger.info("  Loading edges...")
        self.edges = pickle.load(f)

        logger.info("  Loading neurons...")
        self.neurons = pickle.load(f)

        logger.info("  Loading processed files...")
        self.processed_files = pickle.load(f)

        logger.info("  Loading neuron_graphs index...")
        self.neuron_graphs = pickle.load(f)

        # Convert legacy string layer keys to int if needed
        self._migrate_string_keys()

    def _load_v1_msgpack(self, state: dict) -> None:
        """Load v1 msgpack format (single dict with pipe-delimited keys)."""
        self.edges = {}
        for key_str, v in state["edges"].items():
            parts = key_str.split("|")
            layer_src = _legacy_layer_to_int(parts[0])
            layer_tgt = _legacy_layer_to_int(parts[2])
            if layer_src is not None and layer_tgt is not None:
                self.edges[(layer_src, int(parts[1]), layer_tgt, int(parts[3]))] = v

        self.neurons = {}
        for key_str, v in state["neurons"].items():
            parts = key_str.split("|")
            layer = _legacy_layer_to_int(parts[0])
            if layer is not None:
                self.neurons[(layer, int(parts[1]))] = v

        self.processed_files = set(state["processed_files"])

        self.neuron_graphs = {}
        for key_str, v in state.get("neuron_graphs", {}).items():
            parts = key_str.rsplit("_", 1)
            if len(parts) == 2:
                layer = _legacy_layer_to_int(parts[0])
                if layer is not None:
                    self.neuron_graphs[(layer, int(parts[1]))] = [
                        tuple(item) if isinstance(item, list) else item for item in v
                    ]

        self.graphs_processed = state["graphs_processed"]
        self.total_edge_observations = state["total_edge_observations"]

    def _load_v0_pickle(self, state: dict) -> None:
        """Load v0 pickle format (single dict, tuple or list keys)."""
        self.edges = {}
        for k, v in state["edges"].items():
            key = tuple(k) if isinstance(k, list) else k
            # Convert string layers if needed
            if isinstance(key[0], str):
                sl = _legacy_layer_to_int(key[0])
                tl = _legacy_layer_to_int(key[2])
                if sl is not None and tl is not None:
                    self.edges[(sl, key[1], tl, key[3])] = v
            else:
                self.edges[key] = v

        self.neurons = {}
        for k, v in state["neurons"].items():
            key = tuple(k) if isinstance(k, list) else k
            if isinstance(key[0], str):
                layer = _legacy_layer_to_int(key[0])
                if layer is not None:
                    self.neurons[(layer, key[1])] = v
            else:
                self.neurons[key] = v

        self.processed_files = set(state["processed_files"])

        self.neuron_graphs = {}
        for key_str, v in state.get("neuron_graphs", {}).items():
            parts = key_str.rsplit("_", 1)
            if len(parts) == 2:
                layer = _legacy_layer_to_int(parts[0])
                if layer is not None:
                    self.neuron_graphs[(layer, int(parts[1]))] = [
                        tuple(item) if isinstance(item, list) else item for item in v
                    ]

        self.graphs_processed = state["graphs_processed"]
        self.total_edge_observations = state["total_edge_observations"]

    def _migrate_string_keys(self) -> None:
        """Convert any remaining string layer keys to int (for v2 checkpoints
        that were written by the old script with 'L24'-style keys)."""
        if not self.edges:
            return

        sample_key = next(iter(self.edges))
        if isinstance(sample_key[0], int):
            return  # Already int keys

        logger.info("  Migrating string layer keys to int...")
        new_edges: dict[_EdgeKey, _EdgeStats] = {}
        for (sl, sn, tl, tn), stats in self.edges.items():
            sli = _legacy_layer_to_int(sl) if isinstance(sl, str) else sl
            tli = _legacy_layer_to_int(tl) if isinstance(tl, str) else tl
            if sli is not None and tli is not None:
                new_edges[(sli, sn, tli, tn)] = stats
        self.edges = new_edges

        new_neurons: dict[_NeuronKey, int] = {}
        for (layer, neuron), count in self.neurons.items():
            li = _legacy_layer_to_int(layer) if isinstance(layer, str) else layer
            if li is not None:
                new_neurons[(li, neuron)] = count
        self.neurons = new_neurons

        new_ng: dict[_NeuronKey, list[tuple[str, int]]] = {}
        for (layer, neuron), mappings in self.neuron_graphs.items():
            li = _legacy_layer_to_int(layer) if isinstance(layer, str) else layer
            if li is not None:
                new_ng[(li, neuron)] = mappings
        self.neuron_graphs = new_ng

    # ------------------------------------------------------------------
    # Output methods (schema conversions)
    # ------------------------------------------------------------------

    def get_edges(self, min_count: int = 1) -> list[Edge]:
        """Get aggregated edges as Edge schema objects.

        Args:
            min_count: Only return edges seen in at least this many graphs.
        """
        result: list[Edge] = []
        for (sl, sn, tl, tn), stats in self.edges.items():
            count = int(stats[0])
            if count < min_count:
                continue
            result.append(
                Edge(
                    src_layer=sl,
                    src_index=sn,
                    tgt_layer=tl,
                    tgt_index=tn,
                    weight=stats[1] / count,  # mean weight
                    count=count,
                    weight_sum=stats[1],
                    weight_abs_sum=stats[2],
                    weight_sq_sum=stats[3],
                    weight_min=stats[4],
                    weight_max=stats[5],
                    method=ConnectivityMethod.RELP,
                )
            )
        return result

    def get_units(self) -> list[Unit]:
        """Get all observed units as Unit schema objects."""
        return [
            Unit(layer=layer, index=neuron, appearance_count=count)
            for (layer, neuron), count in self.neurons.items()
        ]

    def get_status(self) -> dict[str, Any]:
        """Return status dict with graphs_processed, edges_count, memory estimate."""
        elapsed = time.time() - self._start_time
        rate = self.graphs_processed / elapsed * 3600 if elapsed > 0 else 0

        # ~100 bytes per edge (tuple key ~40 bytes + list value ~60 bytes)
        edge_memory_mb = len(self.edges) * 100 / 1024 / 1024

        return {
            "graphs_processed": self.graphs_processed,
            "edges_count": len(self.edges),
            "neurons_count": len(self.neurons),
            "total_edge_observations": self.total_edge_observations,
            "processed_files_count": len(self.processed_files),
            "elapsed_seconds": elapsed,
            "graphs_per_hour": rate,
            "estimated_memory_mb": edge_memory_mb,
        }

    def get_graphs_for_neuron(
        self, layer: int, neuron: int, limit: int = 100
    ) -> list[dict]:
        """Look up which graphs a neuron appeared in.

        Returns list of dicts with 'graph_file' and 'corpus_index' keys.
        """
        key = (layer, neuron)
        if key not in self.neuron_graphs:
            return []

        mappings = self.neuron_graphs[key][:limit]
        return [
            {"graph_file": graph_file, "corpus_index": corpus_index}
            for graph_file, corpus_index in mappings
        ]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_duckdb(self, db_path: Path) -> None:
        """Export aggregated data to DuckDB database.

        Creates tables: edges, neurons, neuron_graphs, metadata — with indexes.
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required for export: pip install duckdb")

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if db_path.exists():
            db_path.unlink()

        logger.info(f"Exporting to DuckDB: {db_path}")
        conn = duckdb.connect(str(db_path))
        start = time.time()

        # --- edges table ---
        logger.info(f"Creating edges table ({len(self.edges):,} edges)...")
        conn.execute("""
            CREATE TABLE edges (
                src_layer INTEGER,
                src_neuron INTEGER,
                tgt_layer INTEGER,
                tgt_neuron INTEGER,
                count INTEGER,
                weight_sum DOUBLE,
                weight_abs_sum DOUBLE,
                weight_sq_sum DOUBLE,
                weight_min DOUBLE,
                weight_max DOUBLE
            )
        """)

        batch_size = 100_000
        edge_items = list(self.edges.items())
        for i in range(0, len(edge_items), batch_size):
            batch = edge_items[i : i + batch_size]
            values = [
                (sl, sn, tl, tn, int(s[0]), s[1], s[2], s[3], s[4], s[5])
                for (sl, sn, tl, tn), s in batch
            ]
            conn.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
            )
            if (i + batch_size) % 1_000_000 == 0:
                logger.info(f"  Inserted {i + batch_size:,} edges...")

        logger.info("Creating edge indexes...")
        conn.execute("CREATE INDEX idx_edges_src ON edges(src_layer, src_neuron)")
        conn.execute("CREATE INDEX idx_edges_tgt ON edges(tgt_layer, tgt_neuron)")

        # --- neurons table ---
        logger.info(f"Creating neurons table ({len(self.neurons):,} neurons)...")
        conn.execute("""
            CREATE TABLE neurons (
                layer INTEGER,
                neuron INTEGER,
                graph_count INTEGER,
                PRIMARY KEY (layer, neuron)
            )
        """)

        neuron_items = list(self.neurons.items())
        for i in range(0, len(neuron_items), batch_size):
            batch = neuron_items[i : i + batch_size]
            values = [(layer, neuron, count) for (layer, neuron), count in batch]
            conn.executemany("INSERT INTO neurons VALUES (?, ?, ?)", values)

        # --- neuron_graphs table ---
        if self.neuron_graphs:
            logger.info("Creating neuron_graphs table...")
            conn.execute("""
                CREATE TABLE neuron_graphs (
                    layer INTEGER,
                    neuron INTEGER,
                    graph_file VARCHAR,
                    corpus_index INTEGER
                )
            """)

            values = []
            for (layer, neuron), mappings in self.neuron_graphs.items():
                for graph_file, corpus_index in mappings:
                    values.append((layer, neuron, graph_file, corpus_index))
                    if len(values) >= batch_size:
                        conn.executemany(
                            "INSERT INTO neuron_graphs VALUES (?, ?, ?, ?)", values
                        )
                        values = []
            if values:
                conn.executemany(
                    "INSERT INTO neuron_graphs VALUES (?, ?, ?, ?)", values
                )
            conn.execute(
                "CREATE INDEX idx_ng_neuron ON neuron_graphs(layer, neuron)"
            )

        # --- metadata table ---
        conn.execute("""
            CREATE TABLE metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("graphs_processed", str(self.graphs_processed)),
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("total_edge_observations", str(self.total_edge_observations)),
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("export_timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())),
        )

        conn.close()

        elapsed = time.time() - start
        size_mb = db_path.stat().st_size / 1024 / 1024
        logger.info(f"Export complete: {size_mb:.1f}MB in {elapsed:.1f}s")

    def export_to_parquet(self, output_dir: Path) -> None:
        """Export to Parquet files (edges.parquet, neurons.parquet)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for export: pip install pyarrow")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to Parquet: {output_dir}")
        start = time.time()

        # --- edges ---
        logger.info(f"Writing edges.parquet ({len(self.edges):,} edges)...")
        edge_data: dict[str, list] = {
            "src_layer": [],
            "src_neuron": [],
            "tgt_layer": [],
            "tgt_neuron": [],
            "count": [],
            "weight_sum": [],
            "weight_abs_sum": [],
            "weight_sq_sum": [],
            "weight_min": [],
            "weight_max": [],
        }

        for (sl, sn, tl, tn), stats in self.edges.items():
            edge_data["src_layer"].append(sl)
            edge_data["src_neuron"].append(sn)
            edge_data["tgt_layer"].append(tl)
            edge_data["tgt_neuron"].append(tn)
            edge_data["count"].append(int(stats[0]))
            edge_data["weight_sum"].append(stats[1])
            edge_data["weight_abs_sum"].append(stats[2])
            edge_data["weight_sq_sum"].append(stats[3])
            edge_data["weight_min"].append(stats[4])
            edge_data["weight_max"].append(stats[5])

        table = pa.table(edge_data)
        pq.write_table(table, output_dir / "edges.parquet", compression="snappy")

        # --- neurons ---
        logger.info(f"Writing neurons.parquet ({len(self.neurons):,} neurons)...")
        neuron_data: dict[str, list] = {
            "layer": [],
            "neuron": [],
            "graph_count": [],
        }

        for (layer, neuron), count in self.neurons.items():
            neuron_data["layer"].append(layer)
            neuron_data["neuron"].append(neuron)
            neuron_data["graph_count"].append(count)

        table = pa.table(neuron_data)
        pq.write_table(table, output_dir / "neurons.parquet", compression="snappy")

        elapsed = time.time() - start
        edges_size = (output_dir / "edges.parquet").stat().st_size / 1024 / 1024
        neurons_size = (output_dir / "neurons.parquet").stat().st_size / 1024 / 1024
        logger.info(
            f"Export complete: edges={edges_size:.1f}MB, neurons={neurons_size:.1f}MB "
            f"in {elapsed:.1f}s"
        )
