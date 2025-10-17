"""Ensemble registry for tracking which clustering runs belong to which pipeline ensemble.

Uses SQLite to maintain a mapping of (pipeline_run_id, idx, clustering_run_id).
"""

import sqlite3
from contextlib import contextmanager

from spd.settings import SPD_CACHE_DIR

# SQLite database path
_ENSEMBLE_REGISTRY_DB = SPD_CACHE_DIR / "clustering_ensemble_registry.db"


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


def register_clustering_run(pipeline_run_id: str, idx: int, clustering_run_id: str) -> None:
    """Register a clustering run as part of a pipeline ensemble.

    Args:
        pipeline_run_id: The ensemble/pipeline run ID
        idx: Index of this run in the ensemble
        clustering_run_id: The individual clustering run ID
    """
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO ensemble_runs (pipeline_run_id, idx, clustering_run_id) VALUES (?, ?, ?)",
            (pipeline_run_id, idx, clustering_run_id),
        )
        conn.commit()


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
