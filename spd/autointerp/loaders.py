"""Loaders for reading autointerp output files."""

import json
from pathlib import Path

from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


def _load_results_from_jsonl(path: Path) -> dict[str, InterpretationResult]:
    results: dict[str, InterpretationResult] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            result = InterpretationResult(**data)
            results[result.component_key] = result
    return results


def _find_latest_nested_results(autointerp_dir: Path) -> Path | None:
    """Find results.jsonl in the latest timestamped subdirectory."""
    candidates: list[Path] = []
    for subdir in autointerp_dir.iterdir():
        if not subdir.is_dir():
            continue
        # Skip non-run directories (eval, scoring, etc.)
        if subdir.name in ("eval", "scoring"):
            continue
        results_path = subdir / "results.jsonl"
        if results_path.exists():
            candidates.append(results_path)
    if not candidates:
        return None
    # Lexicographic sort on parent dir name (YYYYMMDD_HHMMSS format)
    candidates.sort(key=lambda p: p.parent.name)
    return candidates[-1]


def _find_latest_flat_results(autointerp_dir: Path) -> Path | None:
    """Find latest results_*.jsonl in flat directory (legacy format)."""
    result_files = sorted(autointerp_dir.glob("results_*.jsonl"))
    if not result_files:
        return None
    return result_files[-1]


def find_latest_results_path(wandb_run_id: str) -> Path | None:
    """Find the latest results file, checking nested structure first, then flat."""
    autointerp_dir = get_autointerp_dir(wandb_run_id)
    if not autointerp_dir.exists():
        return None

    # Try nested structure first (new format)
    nested = _find_latest_nested_results(autointerp_dir)
    if nested is not None:
        return nested

    # Fall back to flat structure (legacy format)
    return _find_latest_flat_results(autointerp_dir)


def load_interpretations(
    wandb_run_id: str,
    autointerp_run_id: str | None = None,
) -> dict[str, InterpretationResult] | None:
    """Load interpretation results.

    Args:
        wandb_run_id: The SPD run ID.
        autointerp_run_id: Specific autointerp run to load. If None, loads the latest.
    """
    if autointerp_run_id is not None:
        path = get_autointerp_dir(wandb_run_id) / autointerp_run_id / "results.jsonl"
        if not path.exists():
            return None
        return _load_results_from_jsonl(path)

    path = find_latest_results_path(wandb_run_id)
    if path is None:
        return None
    return _load_results_from_jsonl(path)
