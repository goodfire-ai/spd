"""Cluster mapping endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.state import StateManager
from spd.app.backend.utils import log_errors

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


class ClusterMapping(BaseModel):
    """Cluster mapping from component keys (layer:component_idx) to cluster IDs.

    Singleton clusters (components not grouped with others) have null values.
    """

    mapping: dict[str, int | None]


@router.post("/load")
@log_errors
def load_cluster_mapping(file_path: str) -> ClusterMapping:
    """Load a cluster mapping JSON file from the given path.

    The file should contain a JSON object with:
    - ensemble_id: string
    - notes: string
    - spd_run: wandb path (must match currently loaded run)
    - clusters: dict mapping component keys to cluster IDs
    """
    state = StateManager.get()
    run_state = state.run_state
    if run_state is None:
        raise HTTPException(status_code=400, detail="No run loaded. Load a run first.")

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert "spd_run" in data, "Missing 'spd_run' field in cluster mapping file"
    assert "clusters" in data, "Missing 'clusters' field in cluster mapping file"

    if data["spd_run"] != run_state.run.wandb_path:
        raise HTTPException(
            status_code=400,
            detail=f"Run ID mismatch: cluster file is for '{data['spd_run']}', "
            f"but loaded run is '{run_state.run.wandb_path}'",
        )

    clusters = data["clusters"]
    assert isinstance(clusters, dict), f"Expected 'clusters' to be dict, got {type(clusters)}"
    for key, value in clusters.items():
        assert isinstance(key, str), f"Key must be string, got {type(key)}"
        assert value is None or isinstance(value, int), (
            f"Value must be int or null, got {type(value)}"
        )

    return ClusterMapping(mapping=clusters)
