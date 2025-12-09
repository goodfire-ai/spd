"""SQLite database for local attribution data.

Stores runs, activation contexts, attribution graphs, and component activations.
Attribution graphs can be cached to avoid recomputation.
"""

import gzip
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spd.app.backend.compute import Edge, Node
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    ModelActivationContexts,
    OutputProbability,
)

DEFAULT_DB_PATH = Path.home() / ".spd" / "local_attr.db"


class Run(BaseModel):
    """A run record."""

    id: int
    wandb_path: str


class PromptRecord(BaseModel):
    """A stored prompt record containing token IDs."""

    id: int
    run_id: int
    token_ids: list[int]
    is_custom: bool = False


class OptimizationParams(BaseModel):
    """Optimization parameters that affect graph computation."""

    label_token: int
    imp_min_coeff: float
    ce_loss_coeff: float
    steps: int
    pnorm: float


class OptimizationStats(BaseModel):
    """Statistics from optimized graph computation."""

    label_prob: float
    l0_total: float
    l0_per_layer: dict[str, float]


class StoredGraph(BaseModel):
    """A stored attribution graph."""

    model_config = {"arbitrary_types_allowed": True}

    id: int = -1  # -1 for unsaved graphs, set by DB on save
    edges: list[Edge]
    output_probs: dict[str, OutputProbability]
    optimization_params: OptimizationParams | None = None
    optimization_stats: OptimizationStats | None = None
    ci_lookup: dict[str, float] | None = None  # Optimized CI values (layer:c_idx -> max_ci)


class InterventionRunRecord(BaseModel):
    """A stored intervention run."""

    id: int
    cached_graph_id: int
    selected_nodes: list[str]  # node keys that were selected
    result_json: str  # JSON-encoded InterventionResponse
    created_at: str


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
                ci_lookup_data TEXT,  -- JSON dict of optimized CI values (layer:c_idx -> max_ci)

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

            CREATE TABLE IF NOT EXISTS intervention_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cached_graph_id INTEGER NOT NULL REFERENCES cached_graphs(id),
                selected_nodes TEXT NOT NULL,  -- JSON array of node keys
                result TEXT NOT NULL,  -- JSON InterventionResponse
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_intervention_runs_graph
                ON intervention_runs(cached_graph_id);
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

    def get_activation_contexts_config(
        self, run_id: int
    ) -> ActivationContextsGenerationConfig | None:
        """Get the config used to generate activation contexts for a run."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT config FROM activation_contexts WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None or row["config"] is None:
            return None
        return ActivationContextsGenerationConfig.model_validate(json.loads(row["config"]))

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

    def save_graph(
        self,
        prompt_id: int,
        graph: StoredGraph,
    ) -> None:
        """Save a computed graph for a prompt.

        Args:
            prompt_id: The prompt ID.
            graph: The graph to save.
        """
        conn = self._get_conn()

        # Compress edges and output probs
        edges_json = json.dumps([asdict(e) for e in graph.edges])
        edges_compressed = gzip.compress(edges_json.encode("utf-8"))

        probs_json = json.dumps({k: v.model_dump() for k, v in graph.output_probs.items()})
        probs_compressed = gzip.compress(probs_json.encode("utf-8"))

        is_optimized = 1 if graph.optimization_params else 0

        try:
            if graph.optimization_params:
                assert graph.optimization_stats is not None, (
                    "optimization_stats required for optimized graphs"
                )
                ci_lookup_json = json.dumps(graph.ci_lookup) if graph.ci_lookup else None
                conn.execute(
                    """INSERT INTO cached_graphs
                       (prompt_id, is_optimized,
                        label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm,
                        edges_data, output_probs_data,
                        label_prob, l0_total, l0_per_layer, ci_lookup_data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        prompt_id,
                        is_optimized,
                        graph.optimization_params.label_token,
                        graph.optimization_params.imp_min_coeff,
                        graph.optimization_params.ce_loss_coeff,
                        graph.optimization_params.steps,
                        graph.optimization_params.pnorm,
                        edges_compressed,
                        probs_compressed,
                        graph.optimization_stats.label_prob,
                        graph.optimization_stats.l0_total,
                        json.dumps(graph.optimization_stats.l0_per_layer),
                        ci_lookup_json,
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO cached_graphs
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
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"Graph already cached for prompt_id={prompt_id}. "
                "Use get_graphs() to retrieve existing graph or delete it first."
            ) from e

    def get_graphs(self, prompt_id: int) -> list[StoredGraph]:
        """Retrieve all stored graphs for a prompt.

        Args:
            prompt_id: The prompt ID.

        Returns:
            List of stored graphs (standard and optimized).
        """
        conn = self._get_conn()

        rows = conn.execute(
            """SELECT id, is_optimized, edges_data, output_probs_data,
                      label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm,
                      label_prob, l0_total, l0_per_layer, ci_lookup_data
               FROM cached_graphs
               WHERE prompt_id = ?
               ORDER BY is_optimized, created_at""",
            (prompt_id,),
        ).fetchall()

        def _edge_from_dict(d: dict[str, Any]) -> Edge:
            return Edge(
                source=Node(**d["source"]),
                target=Node(**d["target"]),
                strength=float(d["strength"]),
                is_cross_seq=bool(d["is_cross_seq"]),
            )

        results: list[StoredGraph] = []
        for row in rows:
            edges_json = json.loads(gzip.decompress(row["edges_data"]).decode("utf-8"))
            edges = [_edge_from_dict(e) for e in edges_json]

            probs_json = json.loads(gzip.decompress(row["output_probs_data"]).decode("utf-8"))
            output_probs = {k: OutputProbability(**v) for k, v in probs_json.items()}

            opt_params: OptimizationParams | None = None
            opt_stats: OptimizationStats | None = None

            ci_lookup: dict[str, float] | None = None
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
                if row["ci_lookup_data"]:
                    ci_lookup = json.loads(row["ci_lookup_data"])

            results.append(
                StoredGraph(
                    id=row["id"],
                    edges=edges,
                    output_probs=output_probs,
                    optimization_params=opt_params,
                    optimization_stats=opt_stats,
                    ci_lookup=ci_lookup,
                )
            )

        return results

    def delete_graphs_for_prompt(self, prompt_id: int) -> int:
        """Delete all graphs for a prompt. Returns the number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM cached_graphs WHERE prompt_id = ?", (prompt_id,))
        conn.commit()
        return cursor.rowcount

    def delete_graphs_for_run(self, run_id: int) -> int:
        """Delete all graphs for all prompts in a run. Returns the number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute(
            """DELETE FROM cached_graphs
               WHERE prompt_id IN (SELECT id FROM prompts WHERE run_id = ?)""",
            (run_id,),
        )
        conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------------
    # Intervention run operations
    # -------------------------------------------------------------------------

    def save_intervention_run(
        self,
        graph_id: int,
        selected_nodes: list[str],
        result_json: str,
    ) -> int:
        """Save an intervention run.

        Args:
            graph_id: The cached graph ID this run belongs to.
            selected_nodes: List of node keys that were selected.
            result_json: JSON-encoded InterventionResponse.

        Returns:
            The intervention run ID.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO intervention_runs (cached_graph_id, selected_nodes, result)
               VALUES (?, ?, ?)""",
            (graph_id, json.dumps(selected_nodes), result_json),
        )
        conn.commit()
        run_id = cursor.lastrowid
        assert run_id is not None
        return run_id

    def get_intervention_runs(self, graph_id: int) -> list[InterventionRunRecord]:
        """Get all intervention runs for a graph.

        Args:
            graph_id: The cached graph ID.

        Returns:
            List of intervention run records, ordered by creation time.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, cached_graph_id, selected_nodes, result, created_at
               FROM intervention_runs
               WHERE cached_graph_id = ?
               ORDER BY created_at""",
            (graph_id,),
        ).fetchall()

        return [
            InterventionRunRecord(
                id=row["id"],
                cached_graph_id=row["cached_graph_id"],
                selected_nodes=json.loads(row["selected_nodes"]),
                result_json=row["result"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_intervention_run(self, run_id: int) -> None:
        """Delete an intervention run."""
        conn = self._get_conn()
        conn.execute("DELETE FROM intervention_runs WHERE id = ?", (run_id,))
        conn.commit()

    def delete_intervention_runs_for_graph(self, graph_id: int) -> int:
        """Delete all intervention runs for a graph. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM intervention_runs WHERE cached_graph_id = ?", (graph_id,)
        )
        conn.commit()
        return cursor.rowcount
