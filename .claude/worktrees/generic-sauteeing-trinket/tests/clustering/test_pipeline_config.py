"""Tests for ClusteringPipelineConfig and ClusteringRunConfig with inline config support."""

from pathlib import Path

import pydantic_core
import pytest

from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.merge_config import MergeConfig
from spd.clustering.scripts.run_pipeline import ClusteringPipelineConfig
from spd.settings import REPO_ROOT


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

    def test_error_when_path_does_not_exist(self):
        """Test that error is raised when clustering_run_config_path does not exist."""
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            ClusteringPipelineConfig(
                clustering_run_config_path=Path("nonexistent/path.json"),
                n_runs=2,
                distances_methods=["perm_invariant_hamming"],
                slurm_job_name_prefix=None,
                slurm_partition=None,
                wandb_entity="test",
                create_git_snapshot=False,
            )

    def test_valid_config_with_existing_path(self):
        """Test that config is valid when path points to existing file."""
        expected_path = Path("spd/clustering/configs/crc/resid_mlp1.json")

        config = ClusteringPipelineConfig(
            clustering_run_config_path=expected_path,
            n_runs=2,
            distances_methods=["perm_invariant_hamming"],
            wandb_entity="test",
            create_git_snapshot=False,
        )

        assert config.clustering_run_config_path == expected_path


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
        crc_path = _config.clustering_run_config_path
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
