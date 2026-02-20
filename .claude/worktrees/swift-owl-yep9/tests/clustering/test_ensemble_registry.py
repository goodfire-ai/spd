"""Tests for ensemble_registry module."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from spd.clustering.ensemble_registry import (
    get_clustering_runs,
    register_clustering_run,
)


@pytest.fixture
def _temp_registry_db(monkeypatch: Any):  # pyright: ignore[reportUnusedFunction]
    """Create a temporary registry database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = Path(tmpdir) / "test_registry.db"
        monkeypatch.setattr("spd.clustering.ensemble_registry._ENSEMBLE_REGISTRY_DB", temp_db_path)
        yield temp_db_path


class TestRegisterClusteringRun:
    """Test register_clustering_run() function."""

    def test_register_single_run(self, _temp_registry_db: Any):
        """Test registering a single run."""
        pipeline_id = "pipeline_001"
        run_id = "run_001"

        assigned_idx = register_clustering_run(pipeline_id, run_id)

        # First index should be 0
        assert assigned_idx == 0

        # Verify in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [(0, "run_001")]

    def test_register_multiple_runs(self, _temp_registry_db: Any):
        """Test registering multiple runs sequentially."""
        pipeline_id = "pipeline_002"

        idx0 = register_clustering_run(pipeline_id, "run_001")
        idx1 = register_clustering_run(pipeline_id, "run_002")
        idx2 = register_clustering_run(pipeline_id, "run_003")

        # Should auto-assign 0, 1, 2
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

        # Verify in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [(0, "run_001"), (1, "run_002"), (2, "run_003")]

    def test_different_pipelines_independent(self, _temp_registry_db: Any):
        """Test that different pipelines have independent index sequences."""
        pipeline_a = "pipeline_a"
        pipeline_b = "pipeline_b"

        # Both should start at 0 when auto-assigning
        idx_a0 = register_clustering_run(pipeline_a, "run_a1")
        idx_b0 = register_clustering_run(pipeline_b, "run_b1")

        assert idx_a0 == 0
        assert idx_b0 == 0

        # Both should increment independently
        idx_a1 = register_clustering_run(pipeline_a, "run_a2")
        idx_b1 = register_clustering_run(pipeline_b, "run_b2")

        assert idx_a1 == 1
        assert idx_b1 == 1

        # Verify in database
        runs_a = get_clustering_runs(pipeline_a)
        runs_b = get_clustering_runs(pipeline_b)

        assert runs_a == [(0, "run_a1"), (1, "run_a2")]
        assert runs_b == [(0, "run_b1"), (1, "run_b2")]


class TestGetClusteringRuns:
    """Test get_clustering_runs() function."""

    def test_get_empty_pipeline(self, _temp_registry_db: Any):
        """Test getting runs from a pipeline that doesn't exist."""
        runs = get_clustering_runs("nonexistent_pipeline")
        assert runs == []

    def test_get_runs_sorted_by_index(self, _temp_registry_db: Any):
        """Test that runs are returned sorted by index."""
        pipeline_id = "pipeline_sort"

        # Register runs (indices will be auto-assigned in order)
        register_clustering_run(pipeline_id, "run_000")
        register_clustering_run(pipeline_id, "run_001")
        register_clustering_run(pipeline_id, "run_002")
        register_clustering_run(pipeline_id, "run_003")

        # Should be returned in sorted order
        runs = get_clustering_runs(pipeline_id)
        assert runs == [
            (0, "run_000"),
            (1, "run_001"),
            (2, "run_002"),
            (3, "run_003"),
        ]
