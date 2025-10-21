"""Tests for ClusteringPipelineConfig and ClusteringRunConfig with inline config support."""

from pathlib import Path

import pytest

from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_run_config import ClusteringRunConfig
from spd.clustering.scripts.run_pipeline import ClusteringPipelineConfig
from spd.settings import REPO_ROOT, SPD_CACHE_DIR


class TestClusteringRunConfigStableHash:
    """Test ClusteringRunConfig.stable_hash_b64() method."""

    def test_deterministic_hash(self):
        """Test that stable_hash_b64 is deterministic for identical configs."""
        config1 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )
        config2 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )

        hash1 = config1.stable_hash_b64()
        hash2 = config2.stable_hash_b64()

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes."""
        config1 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )
        config2 = ClusteringRunConfig(
            model_path="wandb:test/project/run2",  # Different model_path
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )
        config3 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=64,  # Different batch_size
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )

        hash1 = config1.stable_hash_b64()
        hash2 = config2.stable_hash_b64()
        hash3 = config3.stable_hash_b64()

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_hash_is_url_safe(self):
        """Test that hash is URL-safe base64 (no padding, URL-safe chars)."""
        config = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )

        hash_value = config.stable_hash_b64()

        # Should not contain padding
        assert "=" not in hash_value

        # Should only contain URL-safe base64 characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert all(c in valid_chars for c in hash_value)

    def test_nested_config_changes_hash(self):
        """Test that changes in nested merge_config affect the hash."""
        config1 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(activation_threshold=0.1),
        )
        config2 = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
            merge_config=MergeConfig(activation_threshold=0.2),  # Different threshold
        )

        assert config1.stable_hash_b64() != config2.stable_hash_b64()


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
                    idx_in_ensemble=0,
                ),
                n_runs=2,
                distances_methods=["perm_invariant_hamming"],
                base_output_dir=Path("/tmp/test"),
                slurm_job_name_prefix=None,
                slurm_partition=None,
                wandb_entity="test",
                create_git_snapshot=False,
            )

    def test_success_with_only_path(self):
        """Test that config validates successfully with only path provided."""
        config = ClusteringPipelineConfig(
            run_clustering_config_path=Path("some/path.json"),
            n_runs=2,
            distances_methods=["perm_invariant_hamming"],
            base_output_dir=Path("/tmp/test"),
            wandb_entity="test",
            create_git_snapshot=False,
        )

        assert config.run_clustering_config_path == Path("some/path.json")
        assert config.run_clustering_config is None

    def test_success_with_only_inline_config(self):
        """Test that config validates successfully with only inline config provided."""
        inline_config = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
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

        assert config.run_clustering_config_path is None
        assert config.run_clustering_config == inline_config


class TestClusteringPipelineConfigGetConfigPath:
    """Test ClusteringPipelineConfig.get_config_path() method."""

    def test_returns_path_directly_when_using_path_field(self):
        """Test that get_config_path returns the path directly when using run_clustering_config_path."""
        expected_path = Path("some/path.json")

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
            idx_in_ensemble=0,
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
        expected_cache_dir = SPD_CACHE_DIR / "merge_run_configs"
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
            idx_in_ensemble=0,
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
            idx_in_ensemble=0,
            merge_config=MergeConfig(),
        )

        # Create a fake collision by manually creating a file with same hash
        hash_value = inline_config.stable_hash_b64()
        cache_dir = SPD_CACHE_DIR / "merge_run_configs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        collision_path = cache_dir / f"{hash_value}.json"

        # Write a different config to the file
        different_config = ClusteringRunConfig(
            model_path="wandb:test/project/run2",  # Different!
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
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

    def test_cache_directory_created_if_not_exists(self):
        """Test that cache directory is created if it doesn't exist."""
        inline_config = ClusteringRunConfig(
            model_path="wandb:test/project/run1",
            batch_size=32,
            dataset_seed=0,
            idx_in_ensemble=0,
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

        cache_dir = SPD_CACHE_DIR / "merge_run_configs"

        # Even if cache dir doesn't exist, get_config_path should create it
        config_path = config.get_config_path()

        assert cache_dir.exists()
        assert config_path.exists()

        # Clean up
        config_path.unlink()


class TestAllConfigsValidation:
    """Test that all existing config files can be loaded and validated."""

    def test_all_pipeline_configs_valid(self):
        """Test that all pipeline config files are valid."""
        configs_dir = REPO_ROOT / "spd" / "clustering" / "configs"

        # Find all YAML/YML files in the configs directory (not subdirectories)
        pipeline_config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

        # Should have at least some configs
        assert len(pipeline_config_files) > 0, "No pipeline config files found"

        errors: list[tuple[Path, Exception]] = []

        for config_file in pipeline_config_files:
            try:
                config = ClusteringPipelineConfig.from_file(config_file)
                # Basic sanity checks
                assert config.n_runs > 0
                assert len(config.distances_methods) > 0
                assert config.wandb_entity is not None
            except Exception as e:
                errors.append((config_file, e))

        # Report all errors at once
        if errors:
            error_msg = "Failed to validate pipeline configs:\n"
            for path, exc in errors:
                error_msg += f"  - {path.name}: {exc}\n"
            pytest.fail(error_msg)

    def test_all_merge_run_configs_valid(self):
        """Test that all merge run config files are valid."""
        mrc_dir = REPO_ROOT / "spd" / "clustering" / "configs" / "mrc"

        # Find all JSON/YAML/YML files in the mrc directory
        mrc_files = (
            list(mrc_dir.glob("*.json"))
            + list(mrc_dir.glob("*.yaml"))
            + list(mrc_dir.glob("*.yml"))
        )

        # Should have at least some configs
        assert len(mrc_files) > 0, "No merge run config files found"

        errors: list[tuple[Path, Exception]] = []

        for config_file in mrc_files:
            try:
                config = ClusteringRunConfig.from_file(config_file)
                # Basic sanity checks
                assert config.batch_size > 0
                assert config.model_path.startswith("wandb:")
                assert config.merge_config is not None
            except Exception as e:
                errors.append((config_file, e))

        # Report all errors at once
        if errors:
            error_msg = "Failed to validate merge run configs:\n"
            for path, exc in errors:
                error_msg += f"  - {path.name}: {exc}\n"
            pytest.fail(error_msg)

    def test_pipeline_configs_reference_valid_mrc_files(self):
        """Test that pipeline configs reference merge run config files that exist."""
        configs_dir = REPO_ROOT / "spd" / "clustering" / "configs"
        pipeline_config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

        errors: list[tuple[Path, str]] = []

        for config_file in pipeline_config_files:
            try:
                config = ClusteringPipelineConfig.from_file(config_file)

                # Skip configs that use inline config
                if config.run_clustering_config is not None:
                    continue

                # Check that referenced file exists
                assert config.run_clustering_config_path is not None
                mrc_path = REPO_ROOT / config.run_clustering_config_path

                if not mrc_path.exists():
                    errors.append(
                        (
                            config_file,
                            f"References non-existent file: {config.run_clustering_config_path}",
                        )
                    )
                else:
                    # Try to load the referenced config
                    try:
                        ClusteringRunConfig.from_file(mrc_path)
                    except Exception as e:
                        errors.append(
                            (
                                config_file,
                                f"Referenced file {mrc_path.name} is invalid: {e}",
                            )
                        )
            except Exception as e:
                errors.append((config_file, f"Failed to load pipeline config: {e}"))

        if errors:
            error_msg = "Pipeline configs with invalid merge run config references:\n"
            for path, msg in errors:
                error_msg += f"  - {path.name}: {msg}\n"
            pytest.fail(error_msg)
