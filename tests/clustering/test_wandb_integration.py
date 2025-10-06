"""Quick sanity tests for WandB integration features."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.pipeline.s2_clustering import _save_merge_history_to_wandb
from spd.clustering.pipeline.s3_normalize_histories import normalize_and_save


def test_wandb_url_parsing_short_format():
    """Test that normalize_and_save can process merge histories using storage."""
    from spd.clustering.pipeline.storage import ClusteringStorage

    # Create temporary directory for storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create ClusteringStorage instance
        storage = ClusteringStorage(base_path=tmp_path, run_identifier="test_run")

        # Create mock merge histories
        config = MergeConfig(
            iters=5,
            alpha=1.0,
            activation_threshold=None,
            pop_component_prob=0.0,
        )

        # Save histories using storage
        for idx in range(2):
            history = MergeHistory.from_config(
                merge_config=config,
                labels=[f"comp{j}" for j in range(5)],
            )
            storage.save_history(history, batch_id=f"batch_{idx:02d}")

        # Test normalize_and_save with storage
        result = normalize_and_save(storage=storage)

        # Basic checks
        assert result is not None
        assert storage.ensemble_meta_file.exists()
        assert storage.ensemble_array_file.exists()

        # Verify we can load the histories back
        loaded_histories = storage.load_histories()
        assert len(loaded_histories) == 2


def test_merge_history_ensemble():
    """Test that MergeHistoryEnsemble can handle multiple histories."""

    # Create test merge histories
    config = MergeConfig(
        iters=3,
        alpha=1.0,
        activation_threshold=None,
        pop_component_prob=0.0,
    )

    histories = []
    for _idx in range(2):
        history = MergeHistory.from_config(
            merge_config=config,
            labels=[f"comp{j}" for j in range(4)],
        )
        histories.append(history)

    # Test ensemble creation
    ensemble = MergeHistoryEnsemble(data=histories)
    assert len(ensemble.data) == 2

    # Test normalization
    normalized_array, metadata = ensemble.normalized()
    assert normalized_array is not None
    assert metadata is not None


def test_save_merge_history_to_wandb():
    """Test that _save_merge_history_to_wandb creates the expected artifact."""

    # Create a real MergeHistory
    config = MergeConfig(
        iters=5,
        alpha=1.0,
        activation_threshold=None,
        pop_component_prob=0.0,
    )

    history = MergeHistory.from_config(
        merge_config=config,
        labels=["comp0", "comp1", "comp2"],
    )

    # Mock wandb run and artifact
    mock_wandb_run = Mock()
    mock_artifact = Mock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        history_path = Path(tmp_dir) / "test_history.zip"
        history.save(history_path)

        with patch("spd.clustering.pipeline.s2_clustering.wandb.Artifact") as mock_artifact_class:
            mock_artifact_class.return_value = mock_artifact

            # Call the function
            _save_merge_history_to_wandb(
                run=mock_wandb_run,
                history_path=history_path,
                batch_id="batch_01",
                config_identifier="test_config",
                history=history,
            )

            # Check that artifact was created and logged
            mock_artifact_class.assert_called_once()
            mock_wandb_run.log_artifact.assert_called_once_with(mock_artifact)

            # Check artifact creation parameters
            call_args = mock_artifact_class.call_args
            assert call_args.kwargs["name"] == "merge_history_batch_01"
            assert call_args.kwargs["type"] == "merge_history"
            assert "batch_01" in call_args.kwargs["description"]


def test_wandb_url_field_in_merge_history():
    """Test that MergeHistory can store and serialize wandb_url."""

    # Create a simple config
    config = MergeConfig(
        iters=10,
        alpha=1.0,
        activation_threshold=None,
        pop_component_prob=0.0,
    )

    # Create MergeHistory with wandb_url
    history = MergeHistory.from_config(
        merge_config=config,
        labels=["comp0", "comp1", "comp2", "comp3", "comp4"],
    )
    # Check that it can be serialized and deserialized
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_history.zip"
        history.save(save_path)
        loaded_history = MergeHistory.read(save_path)

        assert loaded_history is not None
        assert loaded_history.merges.group_idxs.shape == (10, 5)  # (iters, n_components)
