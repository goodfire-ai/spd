"""SQLite database for local attribution data.

Stores runs, activation contexts, attribution graphs, and component activations.
Attribution graphs can be cached to avoid recomputation.
"""

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spd.app.backend.compute import Edge, Node
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    ModelActivationContexts,
    OutputProbability,
    SubcomponentActivationContexts,
    SubcomponentMetadata,
)
from spd.log import logger
from spd.settings import REPO_ROOT

# Persistent data directories
_APP_DATA_DIR = REPO_ROOT / ".data" / "app"
DEFAULT_DB_PATH = _APP_DATA_DIR / "local_attr.db"
CORRELATIONS_DIR = _APP_DATA_DIR / "correlations"


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

    imp_min_coeff: float
    steps: int
    pnorm: float
    # CE loss params (optional, must be set together)
    label_token: int | None = None
    ce_loss_coeff: float | None = None
    # KL loss param (optional)
    kl_loss_coeff: float | None = None


class StoredGraph(BaseModel):
    """A stored attribution graph."""

    model_config = {"arbitrary_types_allowed": True}

    id: int = -1  # -1 for unsaved graphs, set by DB on save
    edges: list[Edge]
    output_probs: dict[str, OutputProbability]  # seq:c_idx -> {prob, token}
    node_ci_vals: dict[str, float]  # layer:seq:c_idx -> ci_val (required for all graphs)
    optimization_params: OptimizationParams | None = None
    label_prob: float | None = (
        None  # P(label_token) with optimized CI mask, only for optimized graphs
    )


class InterventionRunRecord(BaseModel):
    """A stored intervention run."""

    id: int
    graph_id: int
    selected_nodes: list[str]  # node keys that were selected
    result_json: str  # JSON-encoded InterventionResponse
    created_at: str


class ForkedInterventionRunRecord(BaseModel):
    """A forked intervention run with modified tokens."""

    id: int
    intervention_run_id: int
    token_replacements: list[tuple[int, int]]  # [(seq_pos, new_token_id), ...]
    result_json: str  # JSON-encoded InterventionResponse
    created_at: str


class LocalAttrDB:
    """SQLite database for storing and querying local attribution data.

    Schema:
    - runs: One row per SPD run (keyed by wandb_path)
    - activation_contexts: Component metadata + generation config, 1:1 with runs
    - prompts: One row per stored prompt (token sequence), keyed by run_id
    - original_component_seq_max_activations: Inverted index mapping components to prompts by a
      component's max activation for that prompt

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

            -- Metadata per run+context_length
            CREATE TABLE IF NOT EXISTS activation_contexts_meta (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                context_length INTEGER NOT NULL,
                config TEXT,
                PRIMARY KEY (run_id, context_length)
            );

            -- Normalized: one row per component
            CREATE TABLE IF NOT EXISTS component_activation_contexts (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                context_length INTEGER NOT NULL,
                component_key TEXT NOT NULL,  -- "layer:component_idx"
                mean_ci REAL NOT NULL,
                data TEXT NOT NULL,  -- JSON of SubcomponentActivationContexts (without key/mean_ci)
                PRIMARY KEY (run_id, context_length, component_key)
            );

            CREATE INDEX IF NOT EXISTS idx_component_activation_contexts_run
                ON component_activation_contexts(run_id, context_length);

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id),
                token_ids TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                is_custom INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS original_component_seq_max_activations (
                prompt_id INTEGER NOT NULL REFERENCES prompts(id),
                component_key TEXT NOT NULL,
                max_ci REAL NOT NULL,
                positions TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prompts_run_id
                ON prompts(run_id);
            CREATE INDEX IF NOT EXISTS idx_component_key
                ON original_component_seq_max_activations(component_key);
            CREATE INDEX IF NOT EXISTS idx_prompt_id
                ON original_component_seq_max_activations(prompt_id);

            CREATE TABLE IF NOT EXISTS graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL REFERENCES prompts(id),
                is_optimized INTEGER NOT NULL,

                -- Optimization params (NULL for standard graphs)
                label_token INTEGER,
                imp_min_coeff REAL,
                ce_loss_coeff REAL,
                kl_loss_coeff REAL,
                steps INTEGER,
                pnorm REAL,

                -- The actual graph data (JSON)
                edges_data TEXT NOT NULL,
                -- Node CI values: "layer:seq:c_idx" -> ci_val (required for all graphs)
                node_ci_vals TEXT NOT NULL,
                -- Output probabilities: "seq:c_idx" -> {prob, token}
                output_probs_data TEXT NOT NULL,

                -- Optimization stats (NULL for standard graphs)
                label_prob REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_standard
                ON graphs(prompt_id)
                WHERE is_optimized = 0;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_optimized
                ON graphs(prompt_id, label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff, steps, pnorm)
                WHERE is_optimized = 1;

            CREATE INDEX IF NOT EXISTS idx_graphs_prompt
                ON graphs(prompt_id);

            CREATE TABLE IF NOT EXISTS intervention_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id INTEGER NOT NULL REFERENCES graphs(id),
                selected_nodes TEXT NOT NULL,  -- JSON array of node keys
                result TEXT NOT NULL,  -- JSON InterventionResponse
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_intervention_runs_graph
                ON intervention_runs(graph_id);

            CREATE TABLE IF NOT EXISTS forked_intervention_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_run_id INTEGER NOT NULL REFERENCES intervention_runs(id) ON DELETE CASCADE,
                token_replacements TEXT NOT NULL,  -- JSON array of [seq_pos, new_token_id] tuples
                result TEXT NOT NULL,  -- JSON InterventionResponse
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_forked_intervention_runs_parent
                ON forked_intervention_runs(intervention_run_id);
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
    # Component contexts operations
    # -------------------------------------------------------------------------

    def set_component_activation_contexts(
        self,
        run_id: int,
        context_length: int,
        contexts: ModelActivationContexts,
        config: ActivationContextsGenerationConfig | None = None,
    ) -> None:
        """Store activation contexts in normalized form (one row per component)."""
        t0 = time.perf_counter()
        conn = self._get_conn()

        # Store metadata
        config_json = json.dumps(config.model_dump()) if config else None
        conn.execute(
            "INSERT OR REPLACE INTO activation_contexts_meta (run_id, context_length, config) VALUES (?, ?, ?)",
            (run_id, context_length, config_json),
        )

        # Delete existing component rows for this run+context_length
        conn.execute(
            "DELETE FROM component_activation_contexts WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
        )

        # Build rows for insert
        rows: list[tuple[int, int, str, float, str]] = []
        for layer_name, subcomps in contexts.layers.items():
            for subcomp in subcomps:
                component_key = f"{layer_name}:{subcomp.subcomponent_idx}"
                data_dict = subcomp.model_dump(exclude={"subcomponent_idx", "mean_ci"})
                data_json = json.dumps(data_dict)
                rows.append((run_id, context_length, component_key, subcomp.mean_ci, data_json))

        t1 = time.perf_counter()

        conn.executemany(
            """INSERT INTO component_activation_contexts
               (run_id, context_length, component_key, mean_ci, data)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        t2 = time.perf_counter()

        logger.info(
            f"Stored {len(rows)} component contexts: "
            f"prep={1000 * (t1 - t0):.0f}ms, insert={1000 * (t2 - t1):.0f}ms"
        )

    def get_component_activation_contexts_summary(
        self, run_id: int, context_length: int
    ) -> dict[str, list[SubcomponentMetadata]] | None:
        """Get lightweight summary: component_key -> mean_ci (no blob loading)."""
        conn = self._get_conn()

        # Check if data exists
        row = conn.execute(
            "SELECT 1 FROM activation_contexts_meta WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
        ).fetchone()
        if row is None:
            return None

        rows = conn.execute(
            """SELECT component_key, mean_ci FROM component_activation_contexts
               WHERE run_id = ? AND context_length = ?
               ORDER BY mean_ci DESC""",
            (run_id, context_length),
        ).fetchall()

        # Group by layer
        result: dict[str, list[SubcomponentMetadata]] = {}
        for row in rows:
            key = row["component_key"]
            layer, idx_str = key.rsplit(":", 1)
            if layer not in result:
                result[layer] = []
            result[layer].append(
                SubcomponentMetadata(subcomponent_idx=int(idx_str), mean_ci=row["mean_ci"])
            )
        return result

    def get_component_activation_context_detail(
        self, run_id: int, context_length: int, layer: str, component_idx: int
    ) -> SubcomponentActivationContexts | None:
        """Get full data for a single component (fast: only loads one small row)."""
        conn = self._get_conn()
        component_key = f"{layer}:{component_idx}"

        row = conn.execute(
            """SELECT mean_ci, data FROM component_activation_contexts
               WHERE run_id = ? AND context_length = ? AND component_key = ?""",
            (run_id, context_length, component_key),
        ).fetchone()

        if row is None:
            return None

        data_dict = json.loads(row["data"])
        data_dict["subcomponent_idx"] = component_idx
        data_dict["mean_ci"] = row["mean_ci"]

        return SubcomponentActivationContexts.model_validate(data_dict)

    def has_component_activation_contexts(self, run_id: int, context_length: int) -> bool:
        """Check if normalized component contexts exist."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM activation_contexts_meta WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
        ).fetchone()
        return row is not None

    def get_component_activation_contexts_config(
        self, run_id: int, context_length: int
    ) -> ActivationContextsGenerationConfig | None:
        """Get the config from normalized storage."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT config FROM activation_contexts_meta WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
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
                """INSERT INTO original_component_seq_max_activations
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
                """INSERT INTO original_component_seq_max_activations
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
                FROM original_component_seq_max_activations ca
                JOIN prompts p ON ca.prompt_id = p.id
                WHERE p.run_id = ? AND ca.component_key IN ({placeholders})
                GROUP BY ca.prompt_id
                HAVING COUNT(DISTINCT ca.component_key) = ?
            """
            rows = conn.execute(query, (run_id, *component_keys, len(component_keys))).fetchall()
        else:
            query = f"""
                SELECT DISTINCT ca.prompt_id
                FROM original_component_seq_max_activations ca
                JOIN prompts p ON ca.prompt_id = p.id
                WHERE p.run_id = ? AND ca.component_key IN ({placeholders})
            """
            rows = conn.execute(query, (run_id, *component_keys)).fetchall()

        return [row["prompt_id"] for row in rows]

    # -------------------------------------------------------------------------
    # Graph operations
    # -------------------------------------------------------------------------

    def save_graph(
        self,
        prompt_id: int,
        graph: StoredGraph,
    ) -> int:
        """Save a computed graph for a prompt.

        Args:
            prompt_id: The prompt ID.
            graph: The graph to save.

        Returns:
            The database ID of the saved graph.
        """
        conn = self._get_conn()

        edges_json = json.dumps([asdict(e) for e in graph.edges])
        probs_json = json.dumps({k: v.model_dump() for k, v in graph.output_probs.items()})
        node_ci_vals_json = json.dumps(graph.node_ci_vals)
        is_optimized = 1 if graph.optimization_params else 0

        # Extract optimization-specific values (NULL for standard graphs)
        label_token = None
        imp_min_coeff = None
        ce_loss_coeff = None
        kl_loss_coeff = None
        steps = None
        pnorm = None
        label_prob = None

        if graph.optimization_params:
            label_token = graph.optimization_params.label_token
            imp_min_coeff = graph.optimization_params.imp_min_coeff
            ce_loss_coeff = graph.optimization_params.ce_loss_coeff
            kl_loss_coeff = graph.optimization_params.kl_loss_coeff
            steps = graph.optimization_params.steps
            pnorm = graph.optimization_params.pnorm
            label_prob = graph.label_prob  # May be None for KL-only optimization

        try:
            cursor = conn.execute(
                """INSERT INTO graphs
                   (prompt_id, is_optimized,
                    label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff, steps, pnorm,
                    edges_data, output_probs_data, node_ci_vals,
                    label_prob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prompt_id,
                    is_optimized,
                    label_token,
                    imp_min_coeff,
                    ce_loss_coeff,
                    kl_loss_coeff,
                    steps,
                    pnorm,
                    edges_json,
                    probs_json,
                    node_ci_vals_json,
                    label_prob,
                ),
            )
            conn.commit()
            graph_id = cursor.lastrowid
            assert graph_id is not None
            return graph_id
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"Graph already exists for prompt_id={prompt_id}. "
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
            """SELECT id, is_optimized, edges_data, output_probs_data, node_ci_vals,
                      label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff, steps, pnorm,
                      label_prob
               FROM graphs
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
            edges = [_edge_from_dict(e) for e in json.loads(row["edges_data"])]
            output_probs = {
                k: OutputProbability(**v) for k, v in json.loads(row["output_probs_data"]).items()
            }

            node_ci_vals: dict[str, float] = json.loads(row["node_ci_vals"])

            opt_params: OptimizationParams | None = None
            label_prob: float | None = None

            if row["is_optimized"]:
                opt_params = OptimizationParams(
                    imp_min_coeff=row["imp_min_coeff"],
                    steps=row["steps"],
                    pnorm=row["pnorm"],
                    label_token=row["label_token"],
                    ce_loss_coeff=row["ce_loss_coeff"],
                    kl_loss_coeff=row["kl_loss_coeff"],
                )
                label_prob = row["label_prob"]

            results.append(
                StoredGraph(
                    id=row["id"],
                    edges=edges,
                    output_probs=output_probs,
                    node_ci_vals=node_ci_vals,
                    optimization_params=opt_params,
                    label_prob=label_prob,
                )
            )

        return results

    def delete_graphs_for_prompt(self, prompt_id: int) -> int:
        """Delete all graphs for a prompt. Returns the number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM graphs WHERE prompt_id = ?", (prompt_id,))
        conn.commit()
        return cursor.rowcount

    def delete_graphs_for_run(self, run_id: int) -> int:
        """Delete all graphs for all prompts in a run. Returns the number of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute(
            """DELETE FROM graphs
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
            graph_id: The graph ID this run belongs to.
            selected_nodes: List of node keys that were selected.
            result_json: JSON-encoded InterventionResponse.

        Returns:
            The intervention run ID.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO intervention_runs (graph_id, selected_nodes, result)
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
            graph_id: The graph ID.

        Returns:
            List of intervention run records, ordered by creation time.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, graph_id, selected_nodes, result, created_at
               FROM intervention_runs
               WHERE graph_id = ?
               ORDER BY created_at""",
            (graph_id,),
        ).fetchall()

        return [
            InterventionRunRecord(
                id=row["id"],
                graph_id=row["graph_id"],
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
        cursor = conn.execute("DELETE FROM intervention_runs WHERE graph_id = ?", (graph_id,))
        conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------------
    # Forked intervention run operations
    # -------------------------------------------------------------------------

    def save_forked_intervention_run(
        self,
        intervention_run_id: int,
        token_replacements: list[tuple[int, int]],
        result_json: str,
    ) -> int:
        """Save a forked intervention run.

        Args:
            intervention_run_id: The parent intervention run ID.
            token_replacements: List of (seq_pos, new_token_id) tuples.
            result_json: JSON-encoded InterventionResponse.

        Returns:
            The forked intervention run ID.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO forked_intervention_runs (intervention_run_id, token_replacements, result)
               VALUES (?, ?, ?)""",
            (intervention_run_id, json.dumps(token_replacements), result_json),
        )
        conn.commit()
        fork_id = cursor.lastrowid
        assert fork_id is not None
        return fork_id

    def get_forked_intervention_runs(
        self, intervention_run_id: int
    ) -> list[ForkedInterventionRunRecord]:
        """Get all forked runs for an intervention run.

        Args:
            intervention_run_id: The parent intervention run ID.

        Returns:
            List of forked intervention run records, ordered by creation time.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, intervention_run_id, token_replacements, result, created_at
               FROM forked_intervention_runs
               WHERE intervention_run_id = ?
               ORDER BY created_at""",
            (intervention_run_id,),
        ).fetchall()

        return [
            ForkedInterventionRunRecord(
                id=row["id"],
                intervention_run_id=row["intervention_run_id"],
                token_replacements=json.loads(row["token_replacements"]),
                result_json=row["result"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_intervention_run(self, run_id: int) -> InterventionRunRecord | None:
        """Get a single intervention run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, graph_id, selected_nodes, result, created_at
               FROM intervention_runs
               WHERE id = ?""",
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        return InterventionRunRecord(
            id=row["id"],
            graph_id=row["graph_id"],
            selected_nodes=json.loads(row["selected_nodes"]),
            result_json=row["result"],
            created_at=row["created_at"],
        )

    def delete_forked_intervention_run(self, fork_id: int) -> None:
        """Delete a forked intervention run."""
        conn = self._get_conn()
        conn.execute("DELETE FROM forked_intervention_runs WHERE id = ?", (fork_id,))
        conn.commit()
