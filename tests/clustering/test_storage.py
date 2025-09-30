"""Comprehensive tests for ClusteringStorage."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from spd.clustering.math.merge_distances import DistancesMethod
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory
from spd.clustering.merge_run_config import RunConfig
from spd.clustering.pipeline.storage import ClusteringStorage, NormalizedEnsemble


@pytest.fixture
def temp_storage():
    """Create a temporary ClusteringStorage instance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = ClusteringStorage(base_path=Path(tmp_dir), run_identifier="test_run")
        yield storage


@pytest.fixture
def sample_config():
    """Create a sample MergeConfig for testing."""
    return MergeConfig(
        iters=5,
        alpha=1.0,
        activation_threshold=None,
        pop_component_prob=0.0,
    )


class TestStorageInitialization:
    """Test storage initialization and directory structure."""

    def test_storage_creates_run_directory(self):
        """Test that storage creates the run directory on initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            storage = ClusteringStorage(base_path=base_path, run_identifier="test_run")

            assert storage.run_path.exists()
            assert storage.run_path == base_path / "test_run"

    def test_storage_without_run_identifier(self):
        """Test that storage works without a run identifier."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            storage = ClusteringStorage(base_path=base_path, run_identifier=None)

            assert storage.run_path == base_path

    def test_storage_paths_are_consistent(self, temp_storage):
        """Test that all storage paths are under the run path."""
        assert str(temp_storage._dataset_dir).startswith(str(temp_storage.run_path))
        assert str(temp_storage._batches_dir).startswith(str(temp_storage.run_path))
        assert str(temp_storage._histories_dir).startswith(str(temp_storage.run_path))
        assert str(temp_storage._ensemble_dir).startswith(str(temp_storage.run_path))
        assert str(temp_storage._distances_dir).startswith(str(temp_storage.run_path))


class TestRunConfigStorage:
    """Test run configuration storage."""

    def test_save_and_load_run_config(self, temp_storage):
        """Test saving and loading RunConfig."""
        # Create a minimal RunConfig
        config = RunConfig(
            merge_config=MergeConfig(
                iters=10,
                alpha=1.0,
                activation_threshold=None,
                pop_component_prob=0.0,
            ),
            model_path="wandb:entity/project/run_id",
            task_name="lm",
            n_batches=5,
            batch_size=32,
            base_path=temp_storage.base_path,
            workers_per_device=1,
            devices=["cuda"],
        )

        # Save config
        saved_path = temp_storage.save_run_config(config)
        assert saved_path.exists()
        assert saved_path == temp_storage.run_config_file

        # Load and verify
        loaded_config = RunConfig.from_file(saved_path)
        assert loaded_config.n_batches == 5
        assert loaded_config.batch_size == 32
        assert loaded_config.task_name == "lm"


class TestBatchStorage:
    """Test batch data storage."""

    def test_save_single_batch(self, temp_storage):
        """Test saving a single batch."""
        batch = torch.randint(0, 100, (8, 16))  # batch_size=8, seq_len=16
        batch_idx = 0

        saved_path = temp_storage.save_batch(batch, batch_idx)
        assert saved_path.exists()
        assert saved_path.name == "batch_00.npz"

    def test_save_and_load_batch(self, temp_storage):
        """Test saving and loading a batch."""
        original_batch = torch.randint(0, 100, (8, 16))
        batch_idx = 0

        # Save
        temp_storage.save_batch(original_batch, batch_idx)

        # Load
        loaded_batch = temp_storage.load_batch(temp_storage.batch_path(batch_idx))

        # Verify
        assert torch.equal(loaded_batch, original_batch)

    def test_save_multiple_batches(self, temp_storage):
        """Test saving multiple batches using save_batches."""
        batches = [torch.randint(0, 100, (8, 16)) for _ in range(3)]
        config = {"test": "config"}

        saved_paths = temp_storage.save_batches(iter(batches), config)

        assert len(saved_paths) == 3
        assert all(p.exists() for p in saved_paths)
        assert temp_storage.dataset_config_file.exists()

    def test_get_batch_paths(self, temp_storage):
        """Test retrieving all batch paths."""
        # Save some batches
        for i in range(3):
            temp_storage.save_batch(torch.randint(0, 100, (8, 16)), i)

        # Get paths
        batch_paths = temp_storage.get_batch_paths()

        assert len(batch_paths) == 3
        assert all(p.exists() for p in batch_paths)
        # Should be sorted
        assert batch_paths == sorted(batch_paths)


class TestHistoryStorage:
    """Test merge history storage."""

    def test_save_and_load_history(self, temp_storage, sample_config):
        """Test saving and loading merge history."""
        # Create history
        history = MergeHistory.from_config(
            config=sample_config,
            labels=["comp0", "comp1", "comp2"],
        )

        batch_id = "batch_00"

        # Save
        saved_path = temp_storage.save_history(history, batch_id)
        assert saved_path.exists()
        assert "batch_00" in str(saved_path)

        # Load
        loaded_history = temp_storage.load_history(batch_id)
        assert loaded_history is not None
        assert len(loaded_history.labels) == 3

    def test_load_multiple_histories(self, temp_storage, sample_config):
        """Test loading all histories."""
        # Save multiple histories
        for i in range(3):
            history = MergeHistory.from_config(
                config=sample_config,
                labels=[f"comp{j}" for j in range(4)],
            )
            temp_storage.save_history(history, batch_id=f"batch_{i:02d}")

        # Load all
        histories = temp_storage.load_histories()
        assert len(histories) == 3

    def test_get_history_paths(self, temp_storage, sample_config):
        """Test getting all history paths."""
        # Save histories
        for i in range(2):
            history = MergeHistory.from_config(
                config=sample_config,
                labels=["comp0", "comp1"],
            )
            temp_storage.save_history(history, batch_id=f"batch_{i:02d}")

        # Get paths
        history_paths = temp_storage.get_history_paths()
        assert len(history_paths) == 2
        assert all(p.exists() for p in history_paths)


class TestEnsembleStorage:
    """Test ensemble data storage."""

    def test_save_ensemble(self, temp_storage):
        """Test saving ensemble data."""
        # Create dummy ensemble data
        merge_array = np.random.randint(0, 10, size=(2, 5, 8))  # n_ens, n_iters, c_components
        metadata = {"n_ensemble": 2, "n_iters": 5}

        ensemble = NormalizedEnsemble(merge_array=merge_array, metadata=metadata)

        # Save
        meta_path, array_path = temp_storage.save_ensemble(ensemble)

        assert meta_path.exists()
        assert array_path.exists()
        assert meta_path == temp_storage.ensemble_meta_file
        assert array_path == temp_storage.ensemble_array_file

    def test_ensemble_data_integrity(self, temp_storage):
        """Test that ensemble data can be saved and loaded correctly."""
        # Create ensemble data
        original_array = np.random.randint(0, 10, size=(2, 5, 8))
        metadata = {"test": "value", "n_ensemble": 2}

        ensemble = NormalizedEnsemble(merge_array=original_array, metadata=metadata)

        # Save
        _, array_path = temp_storage.save_ensemble(ensemble)

        # Load and verify
        loaded_data = np.load(array_path)
        loaded_array = loaded_data["merges"]

        assert np.array_equal(loaded_array, original_array)


class TestDistancesStorage:
    """Test distance matrix storage."""

    def test_save_distances(self, temp_storage):
        """Test saving distance matrix."""
        distances = np.random.rand(5, 3, 3)  # n_iters, n_ens, n_ens
        method: DistancesMethod = "perm_invariant_hamming"

        saved_path = temp_storage.save_distances(distances, method)

        assert saved_path.exists()
        assert method in saved_path.name

    def test_save_and_load_distances(self, temp_storage):
        """Test saving and loading distances."""
        original_distances = np.random.rand(5, 3, 3)
        method: DistancesMethod = "perm_invariant_hamming"

        # Save
        temp_storage.save_distances(original_distances, method)

        # Load
        loaded_distances = temp_storage.load_distances(method)

        assert np.array_equal(loaded_distances, original_distances)


class TestStorageIntegration:
    """Test integration scenarios."""

    def test_full_pipeline_storage_flow(self, temp_storage, sample_config):
        """Test a complete storage workflow."""
        # 1. Save run config
        run_config = RunConfig(
            merge_config=sample_config,
            model_path="wandb:entity/project/run_id",
            task_name="lm",
            n_batches=2,
            batch_size=8,
            base_path=temp_storage.base_path,
            workers_per_device=1,
            devices=["cpu"],
        )
        temp_storage.save_run_config(run_config)

        # 2. Save batches
        batches = [torch.randint(0, 100, (8, 16)) for _ in range(2)]
        temp_storage.save_batches(iter(batches), {"dataset": "test"})

        # 3. Save histories
        for i in range(2):
            history = MergeHistory.from_config(
                config=sample_config,
                labels=["comp0", "comp1", "comp2"],
            )
            temp_storage.save_history(history, batch_id=f"batch_{i:02d}")

        # 4. Save ensemble
        merge_array = np.random.randint(0, 3, size=(2, 5, 3))
        ensemble = NormalizedEnsemble(
            merge_array=merge_array,
            metadata={"n_ensemble": 2, "n_iters": 5},
        )
        temp_storage.save_ensemble(ensemble)

        # 5. Save distances
        distances = np.random.rand(5, 2, 2)
        temp_storage.save_distances(distances, "perm_invariant_hamming")

        # Verify all files exist
        assert temp_storage.run_config_file.exists()
        assert temp_storage.dataset_config_file.exists()
        assert len(temp_storage.get_batch_paths()) == 2
        assert len(temp_storage.get_history_paths()) == 2
        assert temp_storage.ensemble_meta_file.exists()
        assert temp_storage.ensemble_array_file.exists()

    def test_storage_filesystem_structure(self, temp_storage):
        """Test that the filesystem structure matches documentation."""
        # Create minimal data to generate structure
        temp_storage.save_run_config(
            RunConfig(
                merge_config=MergeConfig(
                    iters=1,
                    alpha=1.0,
                    activation_threshold=None,
                    pop_component_prob=0.0,
                ),
                model_path="wandb:e/p/r",
                task_name="lm",
                n_batches=1,
                batch_size=1,
                base_path=temp_storage.base_path,
                workers_per_device=1,
                devices=["cpu"],
            )
        )

        # Verify structure
        assert (temp_storage.run_path / "run_config.json").exists()

        # The directories are created lazily, so trigger their creation
        temp_storage.save_batch(torch.tensor([[1, 2, 3]]), 0)
        assert (temp_storage.run_path / "dataset" / "batches").exists()
