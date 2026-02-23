"""SQLite database for prompt attribution data.

Stores runs, prompts, and attribution graphs.
Activation contexts and correlations are stored in the harvest pipeline output at
SPD_OUT_DIR/harvest/<run_id>/.
Interpretations are stored separately at SPD_OUT_DIR/autointerp/<run_id>/.
"""

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from spd.app.backend.compute import Edge, Node
from spd.app.backend.optim_cis import MaskType
from spd.app.backend.schemas import OutputProbability
from spd.settings import REPO_ROOT

GraphType = Literal["standard", "optimized", "manual"]

# Persistent data directories
_APP_DATA_DIR = REPO_ROOT / ".data" / "app"
DEFAULT_DB_PATH = _APP_DATA_DIR / "prompt_attr.db"


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
    beta: float
    mask_type: MaskType
    # CE loss params (optional, must be set together)
    label_token: int | None = None
    ce_loss_coeff: float | None = None
    # KL loss param (optional)
    kl_loss_coeff: float | None = None


class StoredGraph(BaseModel):
    """A stored attribution graph."""

    model_config = {"arbitrary_types_allowed": True}

    id: int = -1  # -1 for unsaved graphs, set by DB on save
    graph_type: GraphType = "standard"

    # Core graph data (all types)
    edges: list[Edge]
    out_probs: dict[str, OutputProbability]  # seq:c_idx -> {prob, target_prob, token}
    node_ci_vals: dict[str, float]  # layer:seq:c_idx -> ci_val (required for all graphs)
    node_subcomp_acts: dict[str, float] = {}  # layer:seq:c_idx -> subcomp act (v_i^T @ a)

    # Optimized-specific (None for other types)
    optimization_params: OptimizationParams | None = None
    label_prob: float | None = None  # P(label_token) with optimized CI mask

    # Manual-specific (None for other types)
    included_nodes: list[str] | None = None  # Nodes included in this graph


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


class PromptAttrDB:
    """SQLite database for storing and querying prompt attribution data.

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

    def __enter__(self) -> "PromptAttrDB":
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
                graph_type TEXT NOT NULL,  -- 'standard', 'optimized', 'manual'

                -- Optimization params (NULL for non-optimized graphs)
                label_token INTEGER,
                imp_min_coeff REAL,
                ce_loss_coeff REAL,
                kl_loss_coeff REAL,
                steps INTEGER,
                pnorm REAL,
                beta REAL,
                mask_type TEXT,

                -- Manual graph params (NULL for non-manual graphs)
                included_nodes TEXT,  -- JSON array of node keys in this graph
                included_nodes_hash TEXT,  -- SHA256 hash of sorted JSON for uniqueness

                -- The actual graph data (JSON)
                edges_data TEXT NOT NULL,
                -- Node CI values: "layer:seq:c_idx" -> ci_val (required for all graphs)
                node_ci_vals TEXT NOT NULL,
                -- Node subcomponent activations: "layer:seq:c_idx" -> v_i^T @ a
                node_subcomp_acts TEXT NOT NULL DEFAULT '{}',
                -- Output probabilities: "seq:c_idx" -> {prob, token}
                output_probs_data TEXT NOT NULL,

                -- Optimization stats (NULL for non-optimized graphs)
                label_prob REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- One standard graph per prompt
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_standard
                ON graphs(prompt_id)
                WHERE graph_type = 'standard';

            -- One optimized graph per unique parameter combination
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_optimized
                ON graphs(prompt_id, label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff, steps, pnorm, beta, mask_type)
                WHERE graph_type = 'optimized';

            -- One manual graph per unique node set (using hash for reliable uniqueness)
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_manual
                ON graphs(prompt_id, included_nodes_hash)
                WHERE graph_type = 'manual';

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

    def find_prompt_by_token_ids(
        self,
        run_id: int,
        token_ids: list[int],
        context_length: int,
    ) -> int | None:
        """Find an existing prompt with the same token_ids."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? AND token_ids = ? AND context_length = ?",
            (run_id, json.dumps(token_ids), context_length),
        ).fetchone()
        return row[0] if row else None

    def add_custom_prompt(
        self,
        run_id: int,
        token_ids: list[int],
        active_components: dict[str, tuple[float, list[int]]],
        context_length: int,
    ) -> int:
        """Add a custom prompt to the database, or return existing if duplicate.

        Args:
            run_id: The run this prompt belongs to.
            token_ids: The token IDs for the prompt.
            active_components: Dict mapping component_key to (max_ci, positions).
            context_length: The context length setting.

        Returns:
            The prompt ID (existing or newly created).
        """
        existing_id = self.find_prompt_by_token_ids(run_id, token_ids, context_length)
        if existing_id is not None:
            return existing_id

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
        probs_json = json.dumps({k: v.model_dump() for k, v in graph.out_probs.items()})
        node_ci_vals_json = json.dumps(graph.node_ci_vals)
        node_subcomp_acts_json = json.dumps(graph.node_subcomp_acts)

        # Extract optimization-specific values (NULL for non-optimized graphs)
        label_token = None
        imp_min_coeff = None
        ce_loss_coeff = None
        kl_loss_coeff = None
        steps = None
        pnorm = None
        beta = None
        mask_type = None
        label_prob = None

        if graph.optimization_params:
            label_token = graph.optimization_params.label_token
            imp_min_coeff = graph.optimization_params.imp_min_coeff
            ce_loss_coeff = graph.optimization_params.ce_loss_coeff
            kl_loss_coeff = graph.optimization_params.kl_loss_coeff
            steps = graph.optimization_params.steps
            pnorm = graph.optimization_params.pnorm
            beta = graph.optimization_params.beta
            mask_type = graph.optimization_params.mask_type
            label_prob = graph.label_prob

        # Extract manual-specific values (NULL for non-manual graphs)
        # Sort included_nodes and compute hash for reliable uniqueness
        included_nodes_json: str | None = None
        included_nodes_hash: str | None = None
        if graph.included_nodes:
            included_nodes_json = json.dumps(sorted(graph.included_nodes))
            included_nodes_hash = hashlib.sha256(included_nodes_json.encode()).hexdigest()

        try:
            cursor = conn.execute(
                """INSERT INTO graphs
                   (prompt_id, graph_type,
                    label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff, steps, pnorm,
                    beta, mask_type, included_nodes, included_nodes_hash,
                    edges_data, output_probs_data, node_ci_vals, node_subcomp_acts, label_prob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prompt_id,
                    graph.graph_type,
                    label_token,
                    imp_min_coeff,
                    ce_loss_coeff,
                    kl_loss_coeff,
                    steps,
                    pnorm,
                    beta,
                    mask_type,
                    included_nodes_json,
                    included_nodes_hash,
                    edges_json,
                    probs_json,
                    node_ci_vals_json,
                    node_subcomp_acts_json,
                    label_prob,
                ),
            )
            conn.commit()
            graph_id = cursor.lastrowid
            assert graph_id is not None
            return graph_id
        except sqlite3.IntegrityError as e:
            match graph.graph_type:
                case "standard":
                    raise ValueError(
                        f"Standard graph already exists for prompt_id={prompt_id}. "
                        "Use get_graphs() to retrieve existing graph or delete it first."
                    ) from e
                case "optimized":
                    raise ValueError(
                        f"Optimized graph with same parameters already exists for prompt_id={prompt_id}."
                    ) from e
                case "manual":
                    # Get-or-create semantics: return existing graph ID
                    conn.rollback()
                    row = conn.execute(
                        """SELECT id FROM graphs
                           WHERE prompt_id = ? AND graph_type = 'manual'
                           AND included_nodes_hash = ?""",
                        (prompt_id, included_nodes_hash),
                    ).fetchone()
                    if row:
                        return row["id"]
                    # Should not happen if constraint triggered
                    raise ValueError("A manual graph with the same nodes already exists.") from e

    def _row_to_stored_graph(self, row: sqlite3.Row) -> StoredGraph:
        """Convert a database row to a StoredGraph."""
        edges = [
            Edge(
                source=Node(**e["source"]),
                target=Node(**e["target"]),
                strength=float(e["strength"]),
                is_cross_seq=bool(e["is_cross_seq"]),
            )
            for e in json.loads(row["edges_data"])
        ]
        out_probs = {
            k: OutputProbability(**v) for k, v in json.loads(row["output_probs_data"]).items()
        }
        node_ci_vals: dict[str, float] = json.loads(row["node_ci_vals"])
        node_subcomp_acts: dict[str, float] = json.loads(row["node_subcomp_acts"] or "{}")

        opt_params: OptimizationParams | None = None
        label_prob: float | None = None
        if row["graph_type"] == "optimized":
            opt_params = OptimizationParams(
                imp_min_coeff=row["imp_min_coeff"],
                steps=row["steps"],
                pnorm=row["pnorm"],
                beta=row["beta"],
                mask_type=row["mask_type"],
                label_token=row["label_token"],
                ce_loss_coeff=row["ce_loss_coeff"],
                kl_loss_coeff=row["kl_loss_coeff"],
            )
            label_prob = row["label_prob"]

        # Parse manual-specific fields
        included_nodes: list[str] | None = None
        if row["included_nodes"]:
            included_nodes = json.loads(row["included_nodes"])

        return StoredGraph(
            id=row["id"],
            graph_type=row["graph_type"],
            edges=edges,
            out_probs=out_probs,
            node_ci_vals=node_ci_vals,
            node_subcomp_acts=node_subcomp_acts,
            optimization_params=opt_params,
            label_prob=label_prob,
            included_nodes=included_nodes,
        )

    def get_graphs(self, prompt_id: int) -> list[StoredGraph]:
        """Retrieve all stored graphs for a prompt.

        Args:
            prompt_id: The prompt ID.

        Returns:
            List of stored graphs (standard, optimized, and manual).
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, graph_type, edges_data, output_probs_data, node_ci_vals,
                      node_subcomp_acts, label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff,
                      steps, pnorm, beta, mask_type, label_prob,
                      included_nodes
               FROM graphs
               WHERE prompt_id = ?
               ORDER BY
                   CASE graph_type WHEN 'standard' THEN 0 WHEN 'optimized' THEN 1 ELSE 2 END,
                   created_at""",
            (prompt_id,),
        ).fetchall()
        return [self._row_to_stored_graph(row) for row in rows]

    def get_graph(self, graph_id: int) -> tuple[StoredGraph, int] | None:
        """Retrieve a single graph by its ID. Returns (graph, prompt_id) or None."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, prompt_id, graph_type, edges_data, output_probs_data, node_ci_vals,
                      node_subcomp_acts, label_token, imp_min_coeff, ce_loss_coeff, kl_loss_coeff,
                      steps, pnorm, beta, mask_type, label_prob,
                      included_nodes
               FROM graphs
               WHERE id = ?""",
            (graph_id,),
        ).fetchone()
        if row is None:
            return None
        return (self._row_to_stored_graph(row), row["prompt_id"])

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
