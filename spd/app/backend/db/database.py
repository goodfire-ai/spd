"""SQLite database for local attribution data.

Stores runs, activation contexts, attribution graphs, and component activations.
Attribution graph edges are computed on-demand at serve time, not stored.
"""

import gzip
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    ModelActivationContexts,
)

DEFAULT_DB_PATH = Path.home() / ".spd" / "local_attr.db"


@dataclass
class Run:
    """A run record."""

    id: int
    wandb_path: str


@dataclass
class GraphRecord:
    """An attribution graph record containing token IDs."""

    id: int
    run_id: int
    token_ids: list[int]


class LocalAttrDB:
    """SQLite database for storing and querying local attribution data.

    Schema:
    - runs: One row per SPD run (keyed by wandb_path)
    - activation_contexts: Component metadata + generation config, 1:1 with runs
    - prompts: One row per attribution graph, keyed by run_id
    - component_activations: Inverted index mapping components to graphs

    Attribution graph edges are computed on-demand at serve time, not stored.
    """

    def __init__(self, db_path: Path | None = None, check_same_thread: bool = True):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "LocalAttrDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Schema initialization
    # -------------------------------------------------------------------------

    def init_schema(self) -> None:
        """Initialize the database schema. Safe to call multiple times."""
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                wandb_path TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS activation_contexts (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id),
                data BLOB NOT NULL,
                config TEXT
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id),
                token_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS component_activations (
                prompt_id INTEGER NOT NULL REFERENCES prompts(id),
                component_key TEXT NOT NULL,
                max_ci REAL NOT NULL,
                positions TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prompts_run_id
                ON prompts(run_id);
            CREATE INDEX IF NOT EXISTS idx_component_key
                ON component_activations(component_key);
            CREATE INDEX IF NOT EXISTS idx_prompt_id
                ON component_activations(prompt_id);
        """)
        conn.commit()

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------

    def create_run(self, wandb_path: str) -> int:
        """Create a new run. Returns the run ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO runs (wandb_path) VALUES (?)",
            (wandb_path,),
        )
        conn.commit()
        run_id = cursor.lastrowid
        assert run_id is not None
        return run_id

    def get_run_by_wandb_path(self, wandb_path: str) -> Run | None:
        """Get a run by its wandb path."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, wandb_path FROM runs WHERE wandb_path = ?",
            (wandb_path,),
        ).fetchone()
        if row is None:
            return None
        return Run(id=row["id"], wandb_path=row["wandb_path"])

    def get_run(self, run_id: int) -> Run | None:
        """Get a run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, wandb_path FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return Run(id=row["id"], wandb_path=row["wandb_path"])

    def get_all_runs(self) -> list[Run]:
        """Get all runs in the database."""
        conn = self._get_conn()
        rows = conn.execute("SELECT id, wandb_path FROM runs ORDER BY created_at DESC").fetchall()
        return [Run(id=row["id"], wandb_path=row["wandb_path"]) for row in rows]

    # -------------------------------------------------------------------------
    # Activation contexts operations
    # -------------------------------------------------------------------------

    def get_activation_contexts(self, run_id: int) -> ModelActivationContexts | None:
        """Get the stored activation contexts for a run as a Pydantic model."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM activation_contexts WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        decompressed = gzip.decompress(row["data"])
        json_data = json.loads(decompressed.decode("utf-8"))
        return ModelActivationContexts.model_validate(json_data)

    def get_activation_contexts_raw(self, run_id: int) -> dict[str, Any] | None:
        """Get the stored activation contexts for a run as raw dict."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM activation_contexts WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        decompressed = gzip.decompress(row["data"])
        json_data = json.loads(decompressed.decode("utf-8"))
        assert isinstance(json_data, dict)
        return json_data

    def get_activation_contexts_config(
        self, run_id: int
    ) -> ActivationContextsGenerationConfig | None:
        """Get the generation config used for activation contexts."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT config FROM activation_contexts WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None or row["config"] is None:
            return None
        config_dict = json.loads(row["config"])
        return ActivationContextsGenerationConfig.model_validate(config_dict)

    def set_activation_contexts(
        self,
        run_id: int,
        contexts: ModelActivationContexts,
        config: ActivationContextsGenerationConfig | None = None,
    ) -> None:
        """Store activation contexts for a run."""
        conn = self._get_conn()
        json_bytes = json.dumps(contexts.model_dump()).encode("utf-8")
        compressed = gzip.compress(json_bytes)
        config_json = json.dumps(config.model_dump()) if config else None
        conn.execute(
            "INSERT OR REPLACE INTO activation_contexts (run_id, data, config) VALUES (?, ?, ?)",
            (run_id, compressed, config_json),
        )
        conn.commit()

    def has_activation_contexts(self, run_id: int) -> bool:
        """Check if activation contexts exist for a run."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM activation_contexts WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row is not None

    # -------------------------------------------------------------------------
    # Graph operations
    # -------------------------------------------------------------------------

    def add_graph(
        self,
        run_id: int,
        token_ids: list[int],
        active_components: dict[str, tuple[float, list[int]]] | None = None,
    ) -> int:
        """Add an attribution graph to the database.

        Args:
            run_id: The run this graph belongs to.
            token_ids: List of token IDs for this graph.
            active_components: Optional dict mapping component_key -> (max_ci, positions).

        Returns:
            The graph ID.
        """
        conn = self._get_conn()

        cursor = conn.execute(
            "INSERT INTO prompts (run_id, token_ids) VALUES (?, ?)",
            (run_id, json.dumps(token_ids)),
        )
        graph_id = cursor.lastrowid
        assert graph_id is not None

        if active_components:
            for component_key, (max_ci, positions) in active_components.items():
                conn.execute(
                    """INSERT INTO component_activations
                       (prompt_id, component_key, max_ci, positions)
                       VALUES (?, ?, ?, ?)""",
                    (graph_id, component_key, max_ci, json.dumps(positions)),
                )

        conn.commit()
        return graph_id

    def get_graph(self, graph_id: int) -> GraphRecord | None:
        """Get an attribution graph by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, run_id, token_ids FROM prompts WHERE id = ?",
            (graph_id,),
        ).fetchone()
        if row is None:
            return None

        return GraphRecord(
            id=row["id"],
            run_id=row["run_id"],
            token_ids=json.loads(row["token_ids"]),
        )

    def get_graph_count(self, run_id: int) -> int:
        """Get total number of attribution graphs for a run."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM prompts WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row["cnt"]

    def get_all_graph_ids(self, run_id: int) -> list[int]:
        """Get all graph IDs for a run."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    def has_graphs(self, run_id: int) -> bool:
        """Check if any attribution graphs exist for a run."""
        return self.get_graph_count(run_id) > 0

    # -------------------------------------------------------------------------
    # Query operations
    # -------------------------------------------------------------------------

    def find_graphs_with_components(
        self,
        run_id: int,
        component_keys: list[str],
        require_all: bool = True,
    ) -> list[int]:
        """Find attribution graphs where specified components are active.

        Args:
            run_id: The run to search within.
            component_keys: List of component keys like "h.0.attn.q_proj:5".
            require_all: If True, require ALL components to be active (intersection).
                        If False, require ANY component to be active (union).

        Returns:
            List of graph IDs matching the query.
        """
        assert component_keys, "No component keys provided"

        conn = self._get_conn()
        placeholders = ",".join("?" * len(component_keys))

        if require_all:
            query = f"""
                SELECT ca.prompt_id
                FROM component_activations ca
                JOIN prompts p ON ca.prompt_id = p.id
                WHERE p.run_id = ? AND ca.component_key IN ({placeholders})
                GROUP BY ca.prompt_id
                HAVING COUNT(DISTINCT ca.component_key) = ?
            """
            rows = conn.execute(query, (run_id, *component_keys, len(component_keys))).fetchall()
        else:
            query = f"""
                SELECT DISTINCT ca.prompt_id
                FROM component_activations ca
                JOIN prompts p ON ca.prompt_id = p.id
                WHERE p.run_id = ? AND ca.component_key IN ({placeholders})
            """
            rows = conn.execute(query, (run_id, *component_keys)).fetchall()

        return [row["prompt_id"] for row in rows]

    def get_component_stats(self, run_id: int, component_key: str) -> dict[str, Any]:
        """Get statistics about a component across all graphs in a run."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ca.prompt_id, ca.max_ci, ca.positions
               FROM component_activations ca
               JOIN prompts p ON ca.prompt_id = p.id
               WHERE p.run_id = ? AND ca.component_key = ?""",
            (run_id, component_key),
        ).fetchall()

        if not rows:
            return {"graph_count": 0, "avg_max_ci": 0.0, "graph_ids": []}

        graph_ids = [row["prompt_id"] for row in rows]
        avg_max_ci = sum(row["max_ci"] for row in rows) / len(rows)

        return {
            "graph_count": len(graph_ids),
            "avg_max_ci": avg_max_ci,
            "graph_ids": graph_ids,
        }
