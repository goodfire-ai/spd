"""Tests for ClusteringPipelineConfig and ClusteringRunConfig with inline config support."""

from pathlib import Path

import pytest

from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.merge_config import MergeConfig
from spd.clustering.scripts.run_pipeline import ClusteringPipelineConfig
from spd.settings import REPO_ROOT, SPD_CACHE_DIR


class TestClusteringRunConfigStableHash:
    """Test ClusteringRunConfig.stable_hash_b64() method."""

    def test_stable_hash_b64(self):
        """Test that stable_hash_b64 is deterministic, unique, and URL-safe."""
        # Create 4 configs: 2 identical, 2 different
        config1 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )
        config2 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )
        config3 = ClusteringRunConfig(
            model_path="wandb:test/project/run2",  # Different model_path
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )
        config4 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(
                activation_threshold=0.2
            ),  # Different merge_config to test nested fields
        )

        hash1 = config1.stable_hash_b64()
        hash2 = config2.stable_hash_b64()
        hash3 = config3.stable_hash_b64()
        hash4 = config4.stable_hash_b64()

        # Identical configs produce identical hashes
        assert hash1 == hash2

        # Different configs produce different hashes
        assert hash1 != hash3
        assert hash1 != hash4
        assert hash3 != hash4

        # Hashes are strings
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Hashes are URL-safe base64 (no padding, URL-safe chars only)
        assert "=" not in hash1
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert all(c in valid_chars for c in hash1)


class TestClusteringPipelineConfigValidation:
    """Test ClusteringPipelineConfig validation logic."""

    def test_error_when_neither_field_provided(self):
        """Test that error is raised when neither path nor inline config is provided."""
        with pytest.raises(ValueError, match="Must specify exactly one"):
            ClusteringPipelineConfig(
                n_runs=2,
                distances_methods=["perm_invariant_hamming"],
                base_output_dir=Path("/tmp/test"),
                slurm_job_name_prefix=None,
                slurm_partition=None,
                wandb_entity="test",
                create_git_snapshot=False,
            )

    def test_error_when_both_fields_provided(self):
        """Test that error is raised when both path and inline config are provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            ClusteringPipelineConfig(
                run_clustering_config_path=Path("some/path.json"),
                run_clustering_config=ClusteringRunConfig(
                    model_path="wandb:test/project/run1",
                    batch_size=32,
                    merge_config=MergeConfig(),
                    dataset_seed=0,
                ),
                n_runs=2,
                distances_methods=["perm_invariant_hamming"],
                base_output_dir=Path("/tmp/test"),
                slurm_job_name_prefix=None,
                slurm_partition=None,
                wandb_entity="test",
                create_git_snapshot=False,
            )


class TestClusteringPipelineConfigGetConfigPath:
    """Test ClusteringPipelineConfig.get_config_path() method."""

    def test_returns_path_directly_when_using_path_field(self):
        """Test that get_config_path returns the path directly when using run_clustering_config_path."""
        expected_path = Path("spd/clustering/configs/crc/resid_mlp1.json")

        config = ClusteringPipelineConfig(
            run_clustering_config_path=expected_path,
            n_runs=2,
            distances_methods=["perm_invariant_hamming"],
            base_output_dir=Path("/tmp/test"),
            wandb_entity="test",
            create_git_snapshot=False,
        )

        assert config.get_config_path() == expected_path

    def test_creates_cached_file_when_using_inline_config(self):
        """Test that get_config_path creates a cached file when using inline config."""
        inline_config = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )

        config = ClusteringPipelineConfig(
            run_clustering_config=inline_config,
            n_runs=2,
            distances_methods=["perm_invariant_hamming"],
            base_output_dir=Path("/tmp/test"),
            wandb_entity="test",
            create_git_snapshot=False,
        )

        config_path = config.get_config_path()

        # Check that file exists
        assert config_path.exists()

        # Check that it's in the expected directory
        expected_cache_dir = SPD_CACHE_DIR / "clustering_run_configs"
        assert config_path.parent == expected_cache_dir

        # Check that filename is the hash
        expected_hash = inline_config.stable_hash_b64()
        assert config_path.name == f"{expected_hash}.json"

        # Check that file contents match the config
        loaded_config = ClusteringRunConfig.from_file(config_path)
        assert loaded_config == inline_config

        # Clean up
        config_path.unlink()

    def test_reuses_existing_cached_file(self):
        """Test that get_config_path reuses existing cached file with same hash."""
        inline_config = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )

        config1 = ClusteringPipelineConfig(
            run_clustering_config=inline_config,
            n_runs=2,
            distances_methods=["perm_invariant_hamming"],
            base_output_dir=Path("/tmp/test"),
            wandb_entity="test",
            create_git_snapshot=False,
        )

        # First call creates the file
        config_path1 = config1.get_config_path()
        assert config_path1.exists()

        # Record modification time
        mtime1 = config_path1.stat().st_mtime

        # Create another config with same inline config
        config2 = ClusteringPipelineConfig(
            run_clustering_config=inline_config,
            n_runs=3,  # Different n_runs shouldn't matter
            distances_methods=["perm_invariant_hamming"],
            base_output_dir=Path("/tmp/test"),
            wandb_entity="test",
            create_git_snapshot=False,
        )

        # Second call should reuse the file
        config_path2 = config2.get_config_path()

        assert config_path1 == config_path2
        assert config_path2.stat().st_mtime == mtime1  # File not modified

        # Clean up
        config_path1.unlink()

    def test_hash_collision_detection(self):
        """Test that hash collision is detected when existing file differs."""
        inline_config = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )

        # Create a fake collision by manually creating a file with same hash
        hash_value = inline_config.stable_hash_b64()
        cache_dir = SPD_CACHE_DIR / "clustering_run_configs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        collision_path = cache_dir / f"{hash_value}.json"

        # Write a different config to the file
        different_config = ClusteringRunConfig(
            model_path="wandb:test/project/run2",  # Different!
            batch_size=32,
            dataset_seed=0,
            merge_config=MergeConfig(),
        )
        different_config.to_file(collision_path)

        try:
            config = ClusteringPipelineConfig(
                run_clustering_config=inline_config,
                n_runs=2,
                distances_methods=["perm_invariant_hamming"],
                base_output_dir=Path("/tmp/test"),
                slurm_job_name_prefix=None,
                slurm_partition=None,
                wandb_entity="test",
                create_git_snapshot=False,
            )

            # Should raise ValueError about hash collision
            with pytest.raises(ValueError, match="Hash collision detected"):
                config.get_config_path()
        finally:
            # Clean up
            if collision_path.exists():
                collision_path.unlink()


def _get_config_files(path: Path):
    """Helper to get all config files."""
    pipeline_config_files = (
        list(path.glob("*.yaml")) + list(path.glob("*.yml")) + list(path.glob("*.json"))
    )
    assert len(pipeline_config_files) > 0, f"No pipeline files found in {path}"
    return pipeline_config_files


class TestAllConfigsValidation:
    """Test that all existing config files can be loaded and validated."""

    @pytest.mark.parametrize(
        "config_file",
        _get_config_files(REPO_ROOT / "spd" / "clustering" / "configs"),
        ids=lambda p: p.stem,
    )
    def test_config_validate_pipeline(self, config_file: Path):
        """Test that each pipeline config file is valid."""
        print(config_file)
        _config = ClusteringPipelineConfig.from_file(config_file)
        crc_path = _config.get_config_path()
        print(f"{crc_path = }")
        assert crc_path.exists()

    @pytest.mark.parametrize(
        "config_file",
        _get_config_files(REPO_ROOT / "spd" / "clustering" / "configs" / "crc"),
        ids=lambda p: p.stem,
    )
    def test_config_validate_pipeline_clustering_run(self, config_file: Path):
        """Test that each clustering run config file is valid."""
        print(config_file)
        _config = ClusteringRunConfig.from_file(config_file)
        assert isinstance(_config, ClusteringRunConfig)
