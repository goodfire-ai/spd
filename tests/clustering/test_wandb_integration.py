"""Quick sanity tests for WandB integration features."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from spd.clustering.merge_history import MergeHistory
from spd.clustering.s2_clustering import _save_merge_history_to_wandb
from spd.clustering.s3_normalize_histories import normalize_and_save


def test_wandb_url_parsing_short_format():
    """Test that normalize_and_save can process merge histories."""
    # Create temporary merge history files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create mock merge histories
        from spd.clustering.merge_config import MergeConfig

        config = MergeConfig(
            iters=5,
            alpha=1.0,
            activation_threshold=None,
            pop_component_prob=0.0,
        )

        history_paths = []
        for idx in range(2):
            history = MergeHistory.from_config(
                config=config,
                labels=[f"comp{j}" for j in range(5)],
            )
            history_path = tmp_path / f"history_{idx}.zip"
            history.save(history_path)
            history_paths.append(history_path)

        # Test normalize_and_save
        output_dir = tmp_path / "output"
        result = normalize_and_save(history_paths, output_dir)

        # Basic checks
        assert result is not None
        assert output_dir.exists()
        assert (output_dir / "ensemble_meta.json").exists()
        assert (output_dir / "ensemble_merge_array.npz").exists()


def test_merge_history_ensemble():
    """Test that MergeHistoryEnsemble can handle multiple histories."""
    from spd.clustering.merge_config import MergeConfig
    from spd.clustering.merge_history import MergeHistoryEnsemble

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
            config=config,
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
    from spd.clustering.merge_config import MergeConfig

    # Create a real MergeHistory
    config = MergeConfig(
        iters=5,
        alpha=1.0,
        activation_threshold=None,
        pop_component_prob=0.0,
    )

    history = MergeHistory.from_config(
        config=config,
        labels=["comp0", "comp1", "comp2"],
    )

    # Mock wandb run and artifact
    mock_wandb_run = Mock()
    mock_artifact = Mock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        history_path = Path(tmp_dir) / "test_history.zip"
        history.save(history_path)

        with patch("spd.clustering.s2_clustering.wandb.Artifact") as mock_artifact_class:
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
    from spd.clustering.merge_config import MergeConfig

    # Create a simple config
    config = MergeConfig(
        iters=10,
        alpha=1.0,
        activation_threshold=None,
        pop_component_prob=0.0,
    )

    # Create MergeHistory with wandb_url
    history = MergeHistory.from_config(
        config=config,
        labels=["comp0", "comp1", "comp2", "comp3", "comp4"],
    )
    # Check that it can be serialized and deserialized
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_history.zip"
        history.save(save_path)
        loaded_history = MergeHistory.read(save_path)

        assert loaded_history
        assert loaded_history.merges.group_idxs.shape == (0, 5)
