"""Loaders for reading autointerp output files."""

import json

from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


def load_interpretations(wandb_run_id: str) -> dict[str, InterpretationResult] | None:
    """Load interpretation results from autointerp output."""
    autointerp_dir = get_autointerp_dir(wandb_run_id)
    path = autointerp_dir / "results.jsonl"
    if not path.exists():
        return None

    results: dict[str, InterpretationResult] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            result = InterpretationResult(**data)
            results[result.component_key] = result
    return results
