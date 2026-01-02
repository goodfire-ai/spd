"""Ensemble registry for tracking which clustering runs belong to which pipeline ensemble.

Uses SQLite to maintain a mapping of (pipeline_run_id, idx, clustering_run_id).
"""

import sqlite3
from contextlib import contextmanager

from spd.settings import SPD_OUT_DIR

# SQLite database path
_ENSEMBLE_REGISTRY_DB = SPD_OUT_DIR / "clustering" / "ensemble_registry.db"


@contextmanager
def _get_connection():
    """Context manager for SQLite connection, ensures table exists."""
    _ENSEMBLE_REGISTRY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_ENSEMBLE_REGISTRY_DB)

    try:
        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_runs (
                pipeline_run_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                clustering_run_id TEXT NOT NULL,
                PRIMARY KEY (pipeline_run_id, idx)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pipeline_run_id
            ON ensemble_runs (pipeline_run_id)
        """)
        conn.commit()

        yield conn
    finally:
        conn.close()


def register_clustering_run(pipeline_run_id: str, clustering_run_id: str) -> int:
    """Register a clustering run as part of a pipeline ensemble.

    Args:
        pipeline_run_id: The ensemble/pipeline run ID
        idx: Index of this run in the ensemble. If -1, auto-assigns the next available index.
        clustering_run_id: The individual clustering run ID

    Returns:
        The index assigned to this run (either the provided idx or the auto-assigned one)
    """
    with _get_connection() as conn:
        # Use BEGIN IMMEDIATE for thread-safe auto-increment
        conn.execute("BEGIN IMMEDIATE")

        # Auto-assign next available index, we rely on atomicity of the transaction here
        cursor = conn.execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM ensemble_runs WHERE pipeline_run_id = ?",
            (pipeline_run_id,),
        )
        assigned_idx: int = cursor.fetchone()[0]

        conn.execute(
            "INSERT INTO ensemble_runs (pipeline_run_id, idx, clustering_run_id) VALUES (?, ?, ?)",
            (pipeline_run_id, assigned_idx, clustering_run_id),
        )
        conn.commit()

        return assigned_idx


def get_clustering_runs(pipeline_run_id: str) -> list[tuple[int, str]]:
    """Get all clustering runs for a pipeline ensemble.

    Args:
        pipeline_run_id: The ensemble/pipeline run ID

    Returns:
        List of (idx, clustering_run_id) tuples, sorted by idx
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT idx, clustering_run_id FROM ensemble_runs WHERE pipeline_run_id = ? ORDER BY idx",
            (pipeline_run_id,),
        )
        return cursor.fetchall()
