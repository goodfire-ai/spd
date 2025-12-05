"""SQLite database for local attribution data.

Stores runs, activation contexts, attribution graphs, and component activations.
Attribution graphs can be cached to avoid recomputation.
"""

import gzip
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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
class PromptRecord:
    """A stored prompt record containing token IDs."""

    id: int
    run_id: int
    token_ids: list[int]
    is_custom: bool = False


@dataclass
class CachedEdge:
    """Edge data stored in cache."""

    src: str
    tgt: str
    val: float
    is_cross_seq: bool


@dataclass
class CachedOutputProb:
    """Output probability stored in cache."""

    prob: float
    token: str


@dataclass
class OptimizationParams:
    """Optimization parameters for cache key."""

    label_token: int
    imp_min_coeff: float
    ce_loss_coeff: float
    steps: int
    pnorm: float


@dataclass
class OptimizationStats:
    """Optimization statistics stored in cache."""

    label_prob: float
    l0_total: float
    l0_per_layer: dict[str, float]


@dataclass
class CachedGraph:
    """A cached attribution graph."""

    edges: list[CachedEdge]
    output_probs: dict[str, CachedOutputProb]
    optimization_stats: OptimizationStats | None = None


class LocalAttrDB:
    """SQLite database for storing and querying local attribution data.

    Schema:
    - runs: One row per SPD run (keyed by wandb_path)
    - activation_contexts: Component metadata + generation config, 1:1 with runs
    - prompts: One row per stored prompt (token sequence), keyed by run_id
    - component_activations: Inverted index mapping components to prompts

    Attribution graphs (edges) are computed on-demand at serve time, not stored.
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
                token_ids TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                is_custom INTEGER NOT NULL DEFAULT 0
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

            CREATE TABLE IF NOT EXISTS cached_graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL REFERENCES prompts(id),
                is_optimized INTEGER NOT NULL,

                -- Optimization params (NULL for standard graphs)
                label_token INTEGER,
                imp_min_coeff REAL,
                ce_loss_coeff REAL,
                steps INTEGER,
                pnorm REAL,

                -- Cached data (gzipped JSON)
                edges_data BLOB NOT NULL,
                output_probs_data BLOB NOT NULL,

                -- Optimization stats (NULL for standard graphs)
                label_prob REAL,
                l0_total REAL,
                l0_per_layer TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_cached_graphs_standard
                ON cached_graphs(prompt_id)
                WHERE is_optimized = 0;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_cached_graphs_optimized
                ON cached_graphs(prompt_id, label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm)
                WHERE is_optimized = 1;

            CREATE INDEX IF NOT EXISTS idx_cached_graphs_prompt
                ON cached_graphs(prompt_id);
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
    # Prompt operations
    # -------------------------------------------------------------------------

    def add_prompts(
        self,
        run_id: int,
        prompts: list[tuple[list[int], dict[str, tuple[float, list[int]]]]],
        context_length: int,
    ) -> list[int]:
        """Add multiple prompts to the database in a single transaction.

        Args:
            run_id: The run these prompts belong to.
            prompts: List of (token_ids, active_components) tuples.
            context_length: The context length setting used when generating these prompts.

        Returns:
            List of prompt IDs.
        """
        conn = self._get_conn()
        prompt_ids: list[int] = []
        component_rows: list[tuple[int, str, float, str]] = []

        for token_ids, active_components in prompts:
            cursor = conn.execute(
                "INSERT INTO prompts (run_id, token_ids, context_length) VALUES (?, ?, ?)",
                (run_id, json.dumps(token_ids), context_length),
            )
            prompt_id = cursor.lastrowid
            assert prompt_id is not None
            prompt_ids.append(prompt_id)

            for component_key, (max_ci, positions) in active_components.items():
                component_rows.append((prompt_id, component_key, max_ci, json.dumps(positions)))

        if component_rows:
            conn.executemany(
                """INSERT INTO component_activations
                   (prompt_id, component_key, max_ci, positions) VALUES (?, ?, ?, ?)""",
                component_rows,
            )

        conn.commit()
        return prompt_ids

    def add_custom_prompt(
        self,
        run_id: int,
        token_ids: list[int],
        active_components: dict[str, tuple[float, list[int]]],
        context_length: int,
    ) -> int:
        """Add a custom prompt to the database.

        Args:
            run_id: The run this prompt belongs to.
            token_ids: The token IDs for the prompt.
            active_components: Dict mapping component_key to (max_ci, positions).
            context_length: The context length setting.

        Returns:
            The prompt ID.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO prompts (run_id, token_ids, context_length, is_custom) VALUES (?, ?, ?, 1)",
            (run_id, json.dumps(token_ids), context_length),
        )
        prompt_id = cursor.lastrowid
        assert prompt_id is not None

        component_rows = [
            (prompt_id, component_key, max_ci, json.dumps(positions))
            for component_key, (max_ci, positions) in active_components.items()
        ]
        if component_rows:
            conn.executemany(
                """INSERT INTO component_activations
                   (prompt_id, component_key, max_ci, positions) VALUES (?, ?, ?, ?)""",
                component_rows,
            )

        conn.commit()
        return prompt_id

    def get_prompt(self, prompt_id: int) -> PromptRecord | None:
        """Get a prompt by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, run_id, token_ids, is_custom FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            return None

        return PromptRecord(
            id=row["id"],
            run_id=row["run_id"],
            token_ids=json.loads(row["token_ids"]),
            is_custom=bool(row["is_custom"]),
        )

    def get_prompt_count(self, run_id: int, context_length: int) -> int:
        """Get total number of prompts for a run with a specific context length."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM prompts WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
        ).fetchone()
        return row["cnt"]

    def get_all_prompt_ids(self, run_id: int, context_length: int) -> list[int]:
        """Get all prompt IDs for a run with a specific context length."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? AND context_length = ? ORDER BY id",
            (run_id, context_length),
        ).fetchall()
        return [row["id"] for row in rows]

    def has_prompts(self, run_id: int, context_length: int) -> bool:
        """Check if any prompts exist for a run with a specific context length."""
        return self.get_prompt_count(run_id, context_length) > 0

    # -------------------------------------------------------------------------
    # Query operations
    # -------------------------------------------------------------------------

    def find_prompts_with_components(
        self,
        run_id: int,
        component_keys: list[str],
        require_all: bool = True,
    ) -> list[int]:
        """Find prompts where specified components are active.

        Args:
            run_id: The run to search within.
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

    # -------------------------------------------------------------------------
    # Cached graph operations
    # -------------------------------------------------------------------------

    def get_cached_graph(
        self,
        prompt_id: int,
        optimization_params: OptimizationParams | None = None,
    ) -> CachedGraph | None:
        """Retrieve a cached graph if it exists.

        Args:
            prompt_id: The prompt ID.
            optimization_params: If provided, look for an optimized graph with these params.
                               If None, look for a standard (non-optimized) graph.

        Returns:
            CachedGraph if found, None otherwise.
        """
        conn = self._get_conn()

        if optimization_params is None:
            row = conn.execute(
                """SELECT edges_data, output_probs_data
                   FROM cached_graphs
                   WHERE prompt_id = ? AND is_optimized = 0""",
                (prompt_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT edges_data, output_probs_data, label_prob, l0_total, l0_per_layer
                   FROM cached_graphs
                   WHERE prompt_id = ? AND is_optimized = 1
                     AND label_token = ? AND imp_min_coeff = ? AND ce_loss_coeff = ?
                     AND steps = ? AND pnorm = ?""",
                (
                    prompt_id,
                    optimization_params.label_token,
                    optimization_params.imp_min_coeff,
                    optimization_params.ce_loss_coeff,
                    optimization_params.steps,
                    optimization_params.pnorm,
                ),
            ).fetchone()

        if row is None:
            return None

        # Decompress and parse edges
        edges_json = json.loads(gzip.decompress(row["edges_data"]).decode("utf-8"))
        edges = [CachedEdge(**e) for e in edges_json]

        # Decompress and parse output probs
        probs_json = json.loads(gzip.decompress(row["output_probs_data"]).decode("utf-8"))
        output_probs = {k: CachedOutputProb(**v) for k, v in probs_json.items()}

        # Parse optimization stats if present
        opt_stats = None
        if optimization_params is not None and row["label_prob"] is not None:
            opt_stats = OptimizationStats(
                label_prob=row["label_prob"],
                l0_total=row["l0_total"],
                l0_per_layer=json.loads(row["l0_per_layer"]),
            )

        return CachedGraph(edges=edges, output_probs=output_probs, optimization_stats=opt_stats)

    def save_cached_graph(
        self,
        prompt_id: int,
        edges: list[CachedEdge],
        output_probs: dict[str, CachedOutputProb],
        optimization_params: OptimizationParams | None = None,
        optimization_stats: OptimizationStats | None = None,
    ) -> None:
        """Save a computed graph to the cache.

        Args:
            prompt_id: The prompt ID.
            edges: List of edges (raw, unnormalized).
            output_probs: Dict of output probabilities.
            optimization_params: If provided, save as an optimized graph.
            optimization_stats: Stats from optimization (required if optimization_params provided).
        """
        conn = self._get_conn()

        # Compress edges and output probs
        edges_json = json.dumps([{"src": e.src, "tgt": e.tgt, "val": e.val, "is_cross_seq": e.is_cross_seq} for e in edges])
        edges_compressed = gzip.compress(edges_json.encode("utf-8"))

        probs_json = json.dumps({k: {"prob": v.prob, "token": v.token} for k, v in output_probs.items()})
        probs_compressed = gzip.compress(probs_json.encode("utf-8"))

        is_optimized = 1 if optimization_params else 0

        if optimization_params:
            assert optimization_stats is not None, "optimization_stats required for optimized graphs"
            conn.execute(
                """INSERT OR REPLACE INTO cached_graphs
                   (prompt_id, is_optimized,
                    label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm,
                    edges_data, output_probs_data,
                    label_prob, l0_total, l0_per_layer)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prompt_id,
                    is_optimized,
                    optimization_params.label_token,
                    optimization_params.imp_min_coeff,
                    optimization_params.ce_loss_coeff,
                    optimization_params.steps,
                    optimization_params.pnorm,
                    edges_compressed,
                    probs_compressed,
                    optimization_stats.label_prob,
                    optimization_stats.l0_total,
                    json.dumps(optimization_stats.l0_per_layer),
                ),
            )
        else:
            conn.execute(
                """INSERT OR REPLACE INTO cached_graphs
                   (prompt_id, is_optimized, edges_data, output_probs_data)
                   VALUES (?, ?, ?, ?)""",
                (
                    prompt_id,
                    is_optimized,
                    edges_compressed,
                    probs_compressed,
                ),
            )

        conn.commit()

    def get_all_cached_graphs(
        self,
        prompt_id: int,
    ) -> list[tuple[CachedGraph, OptimizationParams | None]]:
        """Retrieve all cached graphs for a prompt (both standard and optimized).

        Args:
            prompt_id: The prompt ID.

        Returns:
            List of (CachedGraph, OptimizationParams | None) tuples.
            OptimizationParams is None for standard graphs.
        """
        conn = self._get_conn()

        rows = conn.execute(
            """SELECT is_optimized, edges_data, output_probs_data,
                      label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm,
                      label_prob, l0_total, l0_per_layer
               FROM cached_graphs
               WHERE prompt_id = ?
               ORDER BY is_optimized, created_at""",
            (prompt_id,),
        ).fetchall()

        results: list[tuple[CachedGraph, OptimizationParams | None]] = []
        for row in rows:
            edges_json = json.loads(gzip.decompress(row["edges_data"]).decode("utf-8"))
            edges = [CachedEdge(**e) for e in edges_json]

            probs_json = json.loads(gzip.decompress(row["output_probs_data"]).decode("utf-8"))
            output_probs = {k: CachedOutputProb(**v) for k, v in probs_json.items()}

            opt_params: OptimizationParams | None = None
            opt_stats: OptimizationStats | None = None

            if row["is_optimized"]:
                opt_params = OptimizationParams(
                    label_token=row["label_token"],
                    imp_min_coeff=row["imp_min_coeff"],
                    ce_loss_coeff=row["ce_loss_coeff"],
                    steps=row["steps"],
                    pnorm=row["pnorm"],
                )
                opt_stats = OptimizationStats(
                    label_prob=row["label_prob"],
                    l0_total=row["l0_total"],
                    l0_per_layer=json.loads(row["l0_per_layer"]),
                )

            results.append((
                CachedGraph(edges=edges, output_probs=output_probs, optimization_stats=opt_stats),
                opt_params,
            ))

        return results

    def delete_cached_graphs_for_prompt(self, prompt_id: int) -> int:
        """Delete all cached graphs for a prompt. Returns the number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM cached_graphs WHERE prompt_id = ?", (prompt_id,))
        conn.commit()
        return cursor.rowcount

    def delete_cached_graphs_for_run(self, run_id: int) -> int:
        """Delete all cached graphs for all prompts in a run. Returns the number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute(
            """DELETE FROM cached_graphs
               WHERE prompt_id IN (SELECT id FROM prompts WHERE run_id = ?)""",
            (run_id,),
        )
        conn.commit()
        return cursor.rowcount
