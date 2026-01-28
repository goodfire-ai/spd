"""Loaders for reading autointerp output files."""

import json

from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


def load_interpretations(wandb_run_id: str) -> dict[str, InterpretationResult] | None:
    """Load interpretation results from autointerp output (latest timestamped file)."""
    autointerp_dir = get_autointerp_dir(wandb_run_id)

    # Find latest timestamped results file (lexicographic sort works for YYYYMMDD_HHMMSS)
    result_files = sorted(autointerp_dir.glob("results_*.jsonl"))
    if not result_files:
        return None
    path = result_files[-1]

    results: dict[str, InterpretationResult] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            result = InterpretationResult(**data)
            results[result.component_key] = result
    return results
