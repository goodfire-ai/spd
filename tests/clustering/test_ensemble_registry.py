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
def _temp_registry_db(monkeypatch: Any):
    """Create a temporary registry database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = Path(tmpdir) / "test_registry.db"
        monkeypatch.setattr("spd.clustering.ensemble_registry._ENSEMBLE_REGISTRY_DB", temp_db_path)
        yield temp_db_path


class TestRegisterClusteringRun:
    """Test register_clustering_run() function."""

    def test_register_with_explicit_index(self, _temp_registry_db: Any):
        """Test registering a run with an explicit index."""
        pipeline_id = "pipeline_001"
        idx = 0
        run_id = "run_001"

        assigned_idx = register_clustering_run(pipeline_id, idx, run_id)

        # Should return the same index
        assert assigned_idx == idx

        # Verify in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [(0, "run_001")]

    def test_register_multiple_explicit_indices(self, _temp_registry_db: Any):
        """Test registering multiple runs with explicit indices."""
        pipeline_id = "pipeline_002"

        idx0 = register_clustering_run(pipeline_id, 0, "run_001")
        idx1 = register_clustering_run(pipeline_id, 1, "run_002")
        idx2 = register_clustering_run(pipeline_id, 2, "run_003")

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

        # Verify order in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [(0, "run_001"), (1, "run_002"), (2, "run_003")]

    def test_auto_assign_single_index(self, _temp_registry_db: Any):
        """Test auto-assigning a single index."""
        pipeline_id = "pipeline_003"
        run_id = "run_001"

        assigned_idx = register_clustering_run(pipeline_id, -1, run_id)

        # First auto-assigned index should be 0
        assert assigned_idx == 0

        # Verify in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [(0, "run_001")]

    def test_auto_assign_multiple_indices(self, _temp_registry_db: Any):
        """Test auto-assigning multiple indices sequentially."""
        pipeline_id = "pipeline_004"

        idx0 = register_clustering_run(pipeline_id, -1, "run_001")
        idx1 = register_clustering_run(pipeline_id, -1, "run_002")
        idx2 = register_clustering_run(pipeline_id, -1, "run_003")

        # Should auto-assign 0, 1, 2
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

        # Verify in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [(0, "run_001"), (1, "run_002"), (2, "run_003")]

    def test_auto_assign_after_explicit_indices(self, _temp_registry_db: Any):
        """Test that auto-assignment continues from max existing index."""
        pipeline_id = "pipeline_005"

        # Register explicit indices
        register_clustering_run(pipeline_id, 0, "run_001")
        register_clustering_run(pipeline_id, 1, "run_002")

        # Auto-assign should get index 2
        idx = register_clustering_run(pipeline_id, -1, "run_003")
        assert idx == 2

        # Auto-assign again should get index 3
        idx = register_clustering_run(pipeline_id, -1, "run_004")
        assert idx == 3

        # Verify in database
        runs = get_clustering_runs(pipeline_id)
        assert runs == [
            (0, "run_001"),
            (1, "run_002"),
            (2, "run_003"),
            (3, "run_004"),
        ]

    def test_auto_assign_with_gaps(self, _temp_registry_db: Any):
        """Test that auto-assignment uses max+1, even with gaps."""
        pipeline_id = "pipeline_006"

        # Register with gaps: 0, 5, 10
        register_clustering_run(pipeline_id, 0, "run_001")
        register_clustering_run(pipeline_id, 5, "run_002")
        register_clustering_run(pipeline_id, 10, "run_003")

        # Auto-assign should get index 11 (max + 1)
        idx = register_clustering_run(pipeline_id, -1, "run_004")
        assert idx == 11

        # Verify in database (ordered by idx)
        runs = get_clustering_runs(pipeline_id)
        assert runs == [
            (0, "run_001"),
            (5, "run_002"),
            (10, "run_003"),
            (11, "run_004"),
        ]

    def test_mixed_explicit_and_auto_assign(self, _temp_registry_db: Any):
        """Test mixing explicit and auto-assigned indices."""
        pipeline_id = "pipeline_007"

        # Mix of explicit and auto-assigned
        idx0 = register_clustering_run(pipeline_id, -1, "run_001")  # auto: 0
        idx1 = register_clustering_run(pipeline_id, 5, "run_002")  # explicit: 5
        idx2 = register_clustering_run(pipeline_id, -1, "run_003")  # auto: 6
        idx3 = register_clustering_run(pipeline_id, 2, "run_004")  # explicit: 2
        idx4 = register_clustering_run(pipeline_id, -1, "run_005")  # auto: 7

        assert idx0 == 0
        assert idx1 == 5
        assert idx2 == 6
        assert idx3 == 2
        assert idx4 == 7

        # Verify in database (ordered by idx)
        runs = get_clustering_runs(pipeline_id)
        assert runs == [
            (0, "run_001"),
            (2, "run_004"),
            (5, "run_002"),
            (6, "run_003"),
            (7, "run_005"),
        ]

    def test_different_pipelines_independent(self, _temp_registry_db: Any):
        """Test that different pipelines have independent index sequences."""
        pipeline_a = "pipeline_a"
        pipeline_b = "pipeline_b"

        # Both should start at 0 when auto-assigning
        idx_a0 = register_clustering_run(pipeline_a, -1, "run_a1")
        idx_b0 = register_clustering_run(pipeline_b, -1, "run_b1")

        assert idx_a0 == 0
        assert idx_b0 == 0

        # Both should increment independently
        idx_a1 = register_clustering_run(pipeline_a, -1, "run_a2")
        idx_b1 = register_clustering_run(pipeline_b, -1, "run_b2")

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

        # Register out of order
        register_clustering_run(pipeline_id, 5, "run_005")
        register_clustering_run(pipeline_id, 1, "run_001")
        register_clustering_run(pipeline_id, 3, "run_003")
        register_clustering_run(pipeline_id, 0, "run_000")

        # Should be returned in sorted order
        runs = get_clustering_runs(pipeline_id)
        assert runs == [
            (0, "run_000"),
            (1, "run_001"),
            (3, "run_003"),
            (5, "run_005"),
        ]
