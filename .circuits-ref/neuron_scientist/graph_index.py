"""SQLite database for neuron-to-graph index.

This module provides utilities for indexing and querying which RelP graphs
contain specific neurons. The index maps (layer, neuron_idx) -> list of graph paths.
"""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "neuron_graph_index.db"


class GraphIndexDB:
    """SQLite database for neuron-to-graph indexing."""

    def __init__(self, db_path: Path | None = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Uses default if not specified.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        with self.connection() as conn:
            conn.executescript("""
                -- Main index table: maps neurons to graphs
                CREATE TABLE IF NOT EXISTS neuron_graph_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer INTEGER NOT NULL,
                    neuron_idx INTEGER NOT NULL,
                    graph_path TEXT NOT NULL,
                    influence_score REAL,
                    ctx_positions TEXT
                );

                -- Graph metadata table
                CREATE TABLE IF NOT EXISTS graph_metadata (
                    graph_path TEXT PRIMARY KEY,
                    num_nodes INTEGER,
                    num_mlp_neurons INTEGER,
                    prompt_preview TEXT,
                    source TEXT
                );

                -- Build info table
                CREATE TABLE IF NOT EXISTS build_info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            conn.commit()

    def create_indexes(self) -> None:
        """Create indexes for fast lookups. Call after bulk insert."""
        with self.connection() as conn:
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_neuron
                    ON neuron_graph_index(layer, neuron_idx);
                CREATE INDEX IF NOT EXISTS idx_graph_path
                    ON neuron_graph_index(graph_path);
            """)
            conn.commit()

    def insert_neuron_graph(
        self,
        layer: int,
        neuron_idx: int,
        graph_path: str,
        influence_score: float | None = None,
        ctx_positions: list[int] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Insert a single neuron-graph mapping."""
        ctx_json = json.dumps(ctx_positions) if ctx_positions else None

        def _insert(c: sqlite3.Connection):
            c.execute(
                """
                INSERT INTO neuron_graph_index
                    (layer, neuron_idx, graph_path, influence_score, ctx_positions)
                VALUES (?, ?, ?, ?, ?)
                """,
                (layer, neuron_idx, graph_path, influence_score, ctx_json),
            )

        if conn:
            _insert(conn)
        else:
            with self.connection() as c:
                _insert(c)
                c.commit()

    def insert_batch(
        self,
        records: list[tuple[int, int, str, float | None, str | None]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Insert multiple neuron-graph mappings efficiently.

        Args:
            records: List of (layer, neuron_idx, graph_path, influence_score, ctx_positions_json)
        """
        def _insert(c: sqlite3.Connection):
            c.executemany(
                """
                INSERT INTO neuron_graph_index
                    (layer, neuron_idx, graph_path, influence_score, ctx_positions)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )

        if conn:
            _insert(conn)
        else:
            with self.connection() as c:
                _insert(c)
                c.commit()

    def insert_graph_metadata(
        self,
        graph_path: str,
        num_nodes: int,
        num_mlp_neurons: int,
        prompt_preview: str | None = None,
        source: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Insert graph metadata."""
        def _insert(c: sqlite3.Connection):
            c.execute(
                """
                INSERT OR REPLACE INTO graph_metadata
                    (graph_path, num_nodes, num_mlp_neurons, prompt_preview, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (graph_path, num_nodes, num_mlp_neurons, prompt_preview, source),
            )

        if conn:
            _insert(conn)
        else:
            with self.connection() as c:
                _insert(c)
                c.commit()

    def set_build_info(self, key: str, value: str) -> None:
        """Store build metadata."""
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO build_info (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def get_build_info(self, key: str) -> str | None:
        """Retrieve build metadata."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM build_info WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_graphs_for_neuron(
        self,
        layer: int,
        neuron_idx: int,
        limit: int = 100,
        min_influence: float = 0.0,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Find all graphs containing a specific neuron.

        Args:
            layer: Layer number (0-31)
            neuron_idx: Neuron index within layer
            limit: Maximum number of results
            min_influence: Minimum influence score filter
            offset: Offset for pagination

        Returns:
            List of dicts with graph_path, influence_score, ctx_positions
        """
        with self.connection() as conn:
            if min_influence > 0:
                rows = conn.execute(
                    """
                    SELECT graph_path, influence_score, ctx_positions
                    FROM neuron_graph_index
                    WHERE layer = ? AND neuron_idx = ? AND influence_score >= ?
                    ORDER BY influence_score DESC
                    LIMIT ? OFFSET ?
                    """,
                    (layer, neuron_idx, min_influence, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT graph_path, influence_score, ctx_positions
                    FROM neuron_graph_index
                    WHERE layer = ? AND neuron_idx = ?
                    ORDER BY influence_score DESC
                    LIMIT ? OFFSET ?
                    """,
                    (layer, neuron_idx, limit, offset),
                ).fetchall()

            return [
                {
                    "graph_path": row["graph_path"],
                    "influence_score": row["influence_score"],
                    "ctx_positions": (
                        json.loads(row["ctx_positions"])
                        if row["ctx_positions"]
                        else None
                    ),
                }
                for row in rows
            ]

    def get_neuron_frequency(self, layer: int, neuron_idx: int) -> dict[str, Any]:
        """Get frequency statistics for a neuron.

        Args:
            layer: Layer number
            neuron_idx: Neuron index

        Returns:
            Dict with count, avg_influence, max_influence, min_influence
        """
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as count,
                    AVG(influence_score) as avg_influence,
                    MAX(influence_score) as max_influence,
                    MIN(influence_score) as min_influence
                FROM neuron_graph_index
                WHERE layer = ? AND neuron_idx = ?
                """,
                (layer, neuron_idx),
            ).fetchone()

            return {
                "layer": layer,
                "neuron_idx": neuron_idx,
                "neuron_id": f"L{layer}/N{neuron_idx}",
                "graph_count": row["count"],
                "avg_influence": row["avg_influence"],
                "max_influence": row["max_influence"],
                "min_influence": row["min_influence"],
            }

    def find_cooccurring_neurons(
        self,
        layer: int,
        neuron_idx: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find neurons that frequently co-occur with the target neuron.

        Args:
            layer: Target neuron layer
            neuron_idx: Target neuron index
            limit: Maximum results to return

        Returns:
            List of dicts with layer, neuron_idx, cooccurrence_count
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT b.layer, b.neuron_idx, COUNT(*) as cooccur_count
                FROM neuron_graph_index a
                JOIN neuron_graph_index b ON a.graph_path = b.graph_path
                WHERE a.layer = ? AND a.neuron_idx = ?
                  AND NOT (b.layer = ? AND b.neuron_idx = ?)
                GROUP BY b.layer, b.neuron_idx
                ORDER BY cooccur_count DESC
                LIMIT ?
                """,
                (layer, neuron_idx, layer, neuron_idx, limit),
            ).fetchall()

            return [
                {
                    "layer": row["layer"],
                    "neuron_idx": row["neuron_idx"],
                    "neuron_id": f"L{row['layer']}/N{row['neuron_idx']}",
                    "cooccurrence_count": row["cooccur_count"],
                }
                for row in rows
            ]

    def get_graph_metadata(self, graph_path: str) -> dict[str, Any] | None:
        """Get metadata for a specific graph."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM graph_metadata WHERE graph_path = ?",
                (graph_path,),
            ).fetchone()

            if row:
                return dict(row)
            return None

    def get_total_graphs(self) -> int:
        """Get total number of indexed graphs."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT graph_path) as count FROM neuron_graph_index"
            ).fetchone()
            return row["count"]

    def get_total_entries(self) -> int:
        """Get total number of index entries."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM neuron_graph_index"
            ).fetchone()
            return row["count"]

    def get_unique_neurons(self) -> int:
        """Get count of unique neurons in the index."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT layer || '_' || neuron_idx) as count FROM neuron_graph_index"
            ).fetchone()
            return row["count"]


# Convenience functions for tool integration
_default_db: GraphIndexDB | None = None


def get_default_db() -> GraphIndexDB:
    """Get or create the default database instance."""
    global _default_db
    if _default_db is None:
        _default_db = GraphIndexDB()
    return _default_db


def graphs_for_neuron(
    layer: int,
    neuron_idx: int,
    limit: int = 100,
    min_influence: float = 0.0,
) -> list[dict[str, Any]]:
    """Convenience function to find graphs for a neuron."""
    return get_default_db().get_graphs_for_neuron(
        layer, neuron_idx, limit, min_influence
    )


def neuron_frequency(layer: int, neuron_idx: int) -> dict[str, Any]:
    """Convenience function to get neuron frequency stats."""
    return get_default_db().get_neuron_frequency(layer, neuron_idx)


def cooccurring_neurons(
    layer: int, neuron_idx: int, limit: int = 50
) -> list[dict[str, Any]]:
    """Convenience function to find co-occurring neurons."""
    return get_default_db().find_cooccurring_neurons(layer, neuron_idx, limit)
