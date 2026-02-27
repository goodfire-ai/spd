"""Extract cluster mapping from an ensemble at a specific iteration.

Usage:
    python -m spd.clustering.scripts.get_cluster_mapping /path/to/ensemble --n_iterations 299
    python -m spd.clustering.scripts.get_cluster_mapping /path/to/ensemble --n_iterations 299 --run_idx 0
    python -m spd.clustering.scripts.get_cluster_mapping /path/to/ensemble --n_iterations 299 --notes "some notes"

Output format:
    {
        "ensemble_id": "e-5f228e5f",
        "notes": "",
        "spd_run": "spd/goodfire/5cr21lbs",
        "clusters": {"h.0.mlp.down_proj:1": 0, "h.0.mlp.down_proj:2": null, ...}
    }

    Note: Singleton clusters (clusters with only one member) have null values.
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np
import yaml

from spd.settings import REPO_ROOT
from spd.utils.wandb_utils import parse_wandb_run_path


def get_cluster_mapping(
    ensemble_dir: str | Path,
    n_iterations: int,
    run_idx: int = 0,
) -> dict[str, int | None]:
    """Get mapping from component labels to cluster indices at a specific iteration.

    Args:
        ensemble_dir: Path to ensemble directory containing ensemble_merge_array.npz
            and ensemble_meta.json
        n_iterations: Number of iterations to extract clusters from
        run_idx: Run index within the ensemble (default 0)

    Returns:
        Mapping from component label (e.g. "h.0.mlp.down_proj:42") to cluster index,
        or None for singleton clusters (clusters with only one member).
    """
    ensemble_dir = Path(ensemble_dir)

    merge_array_path = ensemble_dir / "ensemble_merge_array.npz"
    meta_path = ensemble_dir / "ensemble_meta.json"

    assert merge_array_path.exists(), f"Merge array not found: {merge_array_path}"
    assert meta_path.exists(), f"Metadata not found: {meta_path}"

    merge_data = np.load(merge_array_path)
    merge_array = merge_data["merge_array"]  # shape: (n_runs, n_iterations, n_components)

    with open(meta_path) as f:
        meta = json.load(f)

    component_labels: list[str] = meta["component_labels"]
    n_runs, n_iterations_stored, n_components = merge_array.shape

    assert 0 <= run_idx < n_runs, f"run_idx {run_idx} out of bounds [0, {n_runs})"
    assert 0 <= n_iterations < n_iterations_stored, (
        f"n_iterations {n_iterations} out of bounds [0, {n_iterations_stored})"
    )
    assert len(component_labels) == n_components, (
        f"Label count mismatch: {len(component_labels)} labels vs {n_components} components"
    )

    assignments = merge_array[run_idx, n_iterations, :]

    # Count members per cluster to identify singletons
    cluster_ids, counts = np.unique(assignments, return_counts=True)
    singleton_clusters = set(cluster_ids[counts == 1])

    return {
        label: None if cluster_id in singleton_clusters else int(cluster_id)
        for label, cluster_id in zip(component_labels, assignments, strict=True)
    }


def get_spd_run_path(ensemble_dir: Path) -> str:
    """Extract the SPD run path from the ensemble's pipeline config.

    Follows pipeline_config.yaml -> clustering_run_config_path -> model_path,
    then parses the wandb path.

    Returns:
        Formatted path like "spd/goodfire/5cr21lbs"
    """
    pipeline_config_path = ensemble_dir / "pipeline_config.yaml"
    assert pipeline_config_path.exists(), f"Pipeline config not found: {pipeline_config_path}"

    with open(pipeline_config_path) as f:
        pipeline_config = yaml.safe_load(f)

    clustering_run_config_path = REPO_ROOT / pipeline_config["clustering_run_config_path"]
    assert clustering_run_config_path.exists(), (
        f"Clustering run config not found: {clustering_run_config_path}"
    )

    with open(clustering_run_config_path) as f:
        clustering_run_config = json.load(f)

    model_path = clustering_run_config["model_path"]
    entity, project, run_id = parse_wandb_run_path(model_path)

    return f"{entity}/{project}/{run_id}"


def main(
    ensemble_dir: str,
    n_iterations: int,
    run_idx: int = 0,
    notes: str = "",
    output: str | None = None,
) -> None:
    """Extract cluster mapping with metadata and output as JSON.

    Args:
        ensemble_dir: Path to ensemble directory
        n_iterations: Number of iterations to extract clusters from
        run_idx: Run index within the ensemble (default 0)
        notes: Optional notes to include in the output
        output: Optional output file path. If not provided, writes to
            {ensemble_dir}/cluster_mapping_{ensemble_id}.json
    """
    ensemble_path = Path(ensemble_dir)

    clusters = get_cluster_mapping(
        ensemble_dir=ensemble_dir,
        n_iterations=n_iterations,
        run_idx=run_idx,
    )

    ensemble_id = ensemble_path.name
    spd_run = get_spd_run_path(ensemble_path)

    result = {
        "ensemble_id": ensemble_id,
        "notes": notes,
        "spd_run": spd_run,
        "n_iterations": n_iterations,
        "run_idx": run_idx,
        "clusters": clusters,
    }

    json_str = json.dumps(result, indent=2)

    if output is None:
        out_path = ensemble_path / f"cluster_mapping_{ensemble_id}.json"
    else:
        out_path = Path(output)

    out_path.write_text(json_str)
    print(f"Wrote mapping ({len(clusters)} components) to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(main)
