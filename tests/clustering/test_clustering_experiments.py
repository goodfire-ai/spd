"""Tests for clustering experiments and notebook-style scripts."""

import subprocess
import sys
from pathlib import Path

import pytest

# Test resource directories
NOTEBOOK_DIR: Path = Path("tests/clustering/scripts")
CONFIG_DIR: Path = Path("spd/clustering/configs")


@pytest.mark.slow
def test_cluster_resid_mlp_notebook():
    """Test running the cluster_resid_mlp.py notebook-style script."""
    script_path = NOTEBOOK_DIR / "cluster_resid_mlp.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    # Run the script as-is
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    # Check that the script ran without errors
    if result.returncode != 0:
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"


@pytest.mark.slow
def test_clustering_with_resid_mlp1_config():
    """Test running clustering with test-resid_mlp1.json config."""
    config_path = CONFIG_DIR / "test-resid_mlp1.json"
    assert config_path.exists(), f"Config not found: {config_path}"

    # Run the clustering main script with the test config
    result = subprocess.run(
        [
            "spd-cluster",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )

    # Check that the script ran without errors
    if result.returncode != 0:
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    assert result.returncode == 0, f"Clustering failed with return code {result.returncode}"


@pytest.mark.slow
def test_cluster_ss_notebook():
    """Test running the cluster_ss.py notebook-style script."""
    script_path = NOTEBOOK_DIR / "cluster_ss.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    # Run the script as-is
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    # Check that the script ran without errors
    if result.returncode != 0:
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"


@pytest.mark.slow
def test_clustering_with_simplestories_config():
    """Test running clustering with test-simplestories.json config."""
    config_path = CONFIG_DIR / "test-simplestories.json"
    assert config_path.exists(), f"Config not found: {config_path}"

    # Run the clustering main script with the test config
    result = subprocess.run(
        [
            "spd-cluster",
            "--config",
            str(config_path),
            "--dataset-streaming",  # see https://github.com/goodfire-ai/spd/pull/199
        ],
        capture_output=True,
        text=True,
    )

    # Check that the script ran without errors
    if result.returncode != 0:
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    assert result.returncode == 0, f"Clustering failed with return code {result.returncode}"
