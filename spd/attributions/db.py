"""SQLite database for local attribution graphs."""

import gzip
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptRecord:
    """A single prompt's attribution data."""

    id: int
    tokens: list[str]
    pairs_json: str  # JSON string of pairs data
    output_probs: dict[str, dict[str, Any]]  # {"s:c": {"prob": float, "token": str}}


@dataclass
class PromptSummary:
    """Lightweight summary of a prompt for search results."""

    id: int
    tokens: list[str]


@dataclass
class PromptRecordSimple:
    """A prompt record for the simplified schema (token IDs only, no pairs)."""

    id: int
    token_ids: list[int]


@dataclass
class ComponentActivation:
    """Record of a component being active in a prompt."""

    prompt_id: int
    component_key: str  # "layer_name:component_idx"
    max_ci: float
    positions: list[int]  # Sequence positions where active


class LocalAttrDB:
    """SQLite database for storing and querying local attribution graphs.

    Schema:
    - meta: Key-value store for model-level data (activation_contexts, config, etc.)
    - prompts: One row per prompt with tokens and gzipped attribution pairs
    - component_activations: Inverted index mapping components to prompts

    Designed for:
    - Single-writer access (use separate DBs for parallel generation, then merge)
    - Fast component-based queries via inverted index
    - Compressed storage for attribution data
    """

    def __init__(self, db_path: Path, check_same_thread: bool = True):
        self.db_path = db_path
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=self._check_same_thread
            )
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

    def init_schema(self) -> None:
        """Initialize the database schema. Safe to call multiple times."""
        conn = self._get_conn()
        # Enable WAL mode for better concurrent write performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tokens TEXT NOT NULL,
                pairs BLOB NOT NULL,
                output_probs BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS component_activations (
                prompt_id INTEGER NOT NULL,
                component_key TEXT NOT NULL,
                max_ci REAL NOT NULL,
                positions TEXT NOT NULL,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_component_key
                ON component_activations(component_key);
            CREATE INDEX IF NOT EXISTS idx_prompt_id
                ON component_activations(prompt_id);
        """)
        conn.commit()

    # -------------------------------------------------------------------------
    # Meta operations
    # -------------------------------------------------------------------------

    def set_meta(self, key: str, value: dict[str, Any] | list[Any]) -> None:
        """Store a JSON-serializable value in the meta table (gzipped)."""
        conn = self._get_conn()
        json_bytes = json.dumps(value).encode("utf-8")
        compressed = gzip.compress(json_bytes)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, compressed),
        )
        conn.commit()

    def get_meta(self, key: str) -> dict[str, Any] | None:
        """Retrieve a value from the meta table."""
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        decompressed = gzip.decompress(row["value"])
        json_data = json.loads(decompressed.decode("utf-8"))
        assert isinstance(json_data, dict)
        return json_data

    def get_activation_contexts(self) -> dict[str, Any] | None:
        """Get the stored activation contexts (model-level component metadata)."""
        return self.get_meta("activation_contexts")

    def set_activation_contexts(self, contexts: dict[str, Any]) -> None:
        """Store activation contexts (computed once per model)."""
        self.set_meta("activation_contexts", contexts)

    # -------------------------------------------------------------------------
    # Prompt operations
    # -------------------------------------------------------------------------

    def add_prompt(
        self,
        tokens: list[str],
        pairs: list[dict[str, Any]],
        active_components: dict[str, ComponentActivation],
        output_probs: dict[str, dict[str, Any]],
    ) -> int:
        """Add a prompt to the database.

        Args:
            tokens: List of token strings for this prompt.
            pairs: List of pair attribution dicts (sparse format).
            active_components: Dict mapping component_key -> ComponentActivation.
            output_probs: Dict mapping "s:c" -> {"prob": float, "token": str}.

        Returns:
            The prompt ID.
        """
        conn = self._get_conn()

        # Compress pairs data
        pairs_json = json.dumps(pairs)
        pairs_compressed = gzip.compress(pairs_json.encode("utf-8"))

        # Compress output_probs data
        output_probs_json = json.dumps(output_probs)
        output_probs_compressed = gzip.compress(output_probs_json.encode("utf-8"))

        # Insert prompt
        cursor = conn.execute(
            "INSERT INTO prompts (tokens, pairs, output_probs) VALUES (?, ?, ?)",
            (json.dumps(tokens), pairs_compressed, output_probs_compressed),
        )
        prompt_id = cursor.lastrowid
        assert prompt_id is not None

        # Insert component activations for inverted index
        for component_key, activation in active_components.items():
            conn.execute(
                """INSERT INTO component_activations
                   (prompt_id, component_key, max_ci, positions)
                   VALUES (?, ?, ?, ?)""",
                (
                    prompt_id,
                    component_key,
                    activation.max_ci,
                    json.dumps(activation.positions),
                ),
            )

        conn.commit()
        return prompt_id

    def get_prompt(self, prompt_id: int) -> PromptRecord | None:
        """Get a single prompt by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, tokens, pairs, output_probs FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            return None

        pairs_decompressed = gzip.decompress(row["pairs"]).decode("utf-8")
        output_probs_decompressed = gzip.decompress(row["output_probs"]).decode("utf-8")
        return PromptRecord(
            id=row["id"],
            tokens=json.loads(row["tokens"]),
            pairs_json=pairs_decompressed,
            output_probs=json.loads(output_probs_decompressed),
        )

    def get_prompt_count(self) -> int:
        """Get total number of prompts in the database."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM prompts").fetchone()
        return row["cnt"]

    def get_all_prompt_summaries(self) -> list[PromptSummary]:
        """Get lightweight summaries of all prompts."""
        conn = self._get_conn()
        rows = conn.execute("SELECT id, tokens FROM prompts ORDER BY id").fetchall()
        return [
            PromptSummary(id=row["id"], tokens=json.loads(row["tokens"])) for row in rows
        ]

    # -------------------------------------------------------------------------
    # Query operations
    # -------------------------------------------------------------------------

    def find_prompts_with_components(
        self,
        component_keys: list[str],
        require_all: bool = True,
    ) -> list[int]:
        """Find prompts where specified components are active.

        Args:
            component_keys: List of component keys like "h.0.attn.q_proj:5".
            require_all: If True, require ALL components to be active (intersection).
                        If False, require ANY component to be active (union).

        Returns:
            List of prompt IDs matching the query.
        """
        assert component_keys, "No component keys provided"

        conn = self._get_conn()
        placeholders = ",".join("?" * len(component_keys))

        if require_all:
            # Intersection: prompts where all components appear
            query = f"""
                SELECT prompt_id
                FROM component_activations
                WHERE component_key IN ({placeholders})
                GROUP BY prompt_id
                HAVING COUNT(DISTINCT component_key) = ?
            """
            rows = conn.execute(query, (*component_keys, len(component_keys))).fetchall()
        else:
            # Union: prompts where any component appears
            query = f"""
                SELECT DISTINCT prompt_id
                FROM component_activations
                WHERE component_key IN ({placeholders})
            """
            rows = conn.execute(query, component_keys).fetchall()

        return [row["prompt_id"] for row in rows]

    def get_component_stats(self, component_key: str) -> dict[str, Any]:
        """Get statistics about a component across all prompts.

        Returns:
            Dict with: prompt_count, avg_max_ci, prompt_ids
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT prompt_id, max_ci, positions
               FROM component_activations
               WHERE component_key = ?""",
            (component_key,),
        ).fetchall()

        if not rows:
            return {"prompt_count": 0, "avg_max_ci": 0.0, "prompt_ids": []}

        prompt_ids = [row["prompt_id"] for row in rows]
        avg_max_ci = sum(row["max_ci"] for row in rows) / len(rows)

        return {
            "prompt_count": len(prompt_ids),
            "avg_max_ci": avg_max_ci,
            "prompt_ids": prompt_ids,
        }

    def get_unique_components(self) -> list[str]:
        """Get all unique component keys in the database."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT component_key FROM component_activations ORDER BY component_key"
        ).fetchall()
        return [row["component_key"] for row in rows]

    # -------------------------------------------------------------------------
    # Merge operations (for parallel generation)
    # -------------------------------------------------------------------------

    def merge_from(self, other_db_path: Path) -> int:
        """Merge prompts from another database into this one.

        Note: Does NOT merge meta table (activation_contexts should be set once).

        Args:
            other_db_path: Path to the database to merge from.

        Returns:
            Number of prompts merged.
        """
        conn = self._get_conn()

        # Attach the other database
        conn.execute("ATTACH DATABASE ? AS other", (str(other_db_path),))

        # Get count before
        count_before = self.get_prompt_count()

        # Copy prompts (IDs will be reassigned)
        conn.execute("""
            INSERT INTO prompts (tokens, pairs, output_probs)
            SELECT tokens, pairs, output_probs FROM other.prompts
        """)

        # Get the ID offset (first new ID)
        id_offset = count_before

        # Copy component_activations with adjusted prompt_ids
        conn.execute(
            """
            INSERT INTO component_activations (prompt_id, component_key, max_ci, positions)
            SELECT prompt_id + ?, component_key, max_ci, positions
            FROM other.component_activations
        """,
            (id_offset,),
        )

        conn.execute("DETACH DATABASE other")
        conn.commit()

        count_after = self.get_prompt_count()
        return count_after - count_before

    # -------------------------------------------------------------------------
    # Simplified schema operations (CI-only storage, on-demand graph computation)
    # -------------------------------------------------------------------------

    def init_schema_simple(self) -> None:
        """Initialize the simplified schema (token IDs only, no pairs).

        This schema stores just token IDs and the inverted index.
        Attribution graphs are computed on-demand at serve time.
        """
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS component_activations (
                prompt_id INTEGER NOT NULL,
                component_key TEXT NOT NULL,
                max_ci REAL NOT NULL,
                positions TEXT NOT NULL,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_component_key
                ON component_activations(component_key);
            CREATE INDEX IF NOT EXISTS idx_prompt_id
                ON component_activations(prompt_id);
        """)
        conn.commit()

    def add_prompt_simple(
        self,
        token_ids: list[int],
        active_components: dict[str, tuple[float, list[int]]],
    ) -> int:
        """Add a prompt with token IDs only (no pairs).

        Args:
            token_ids: List of token IDs for this prompt.
            active_components: Dict mapping component_key -> (max_ci, positions).

        Returns:
            The prompt ID.
        """
        conn = self._get_conn()

        cursor = conn.execute(
            "INSERT INTO prompts (token_ids) VALUES (?)",
            (json.dumps(token_ids),),
        )
        prompt_id = cursor.lastrowid
        assert prompt_id is not None

        for component_key, (max_ci, positions) in active_components.items():
            conn.execute(
                """INSERT INTO component_activations
                   (prompt_id, component_key, max_ci, positions)
                   VALUES (?, ?, ?, ?)""",
                (prompt_id, component_key, max_ci, json.dumps(positions)),
            )

        conn.commit()
        return prompt_id

    def get_prompt_simple(self, prompt_id: int) -> PromptRecordSimple | None:
        """Get a prompt's token IDs by ID (simplified schema)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, token_ids FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            return None

        return PromptRecordSimple(
            id=row["id"],
            token_ids=json.loads(row["token_ids"]),
        )

    def get_all_prompt_ids(self) -> list[int]:
        """Get all prompt IDs in the database."""
        conn = self._get_conn()
        rows = conn.execute("SELECT id FROM prompts ORDER BY id").fetchall()
        return [row["id"] for row in rows]
