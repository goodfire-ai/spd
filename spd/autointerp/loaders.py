"""Loaders for reading autointerp output files."""

import json
from pathlib import Path

from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir
from spd.harvest.schemas import get_harvest_dir


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


def _load_scores_from_jsonl(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            scores[data["component_key"]] = data["score"]
    return scores


def _find_latest_jsonl(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("results_*.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def load_intruder_scores(run_id: str) -> dict[str, float] | None:
    """Load intruder eval scores. Checks harvest dir first, then legacy autointerp dir."""
    # Harvest location (current)
    harvest_path = _find_latest_jsonl(get_harvest_dir(run_id) / "eval" / "intruder")
    if harvest_path is not None:
        return _load_scores_from_jsonl(harvest_path)

    # Legacy: autointerp/{run_id}/scoring/intruder/
    legacy_path = _find_latest_jsonl(get_autointerp_dir(run_id) / "scoring" / "intruder")
    if legacy_path is not None:
        return _load_scores_from_jsonl(legacy_path)

    return None


def _find_latest_autointerp_run_dir(run_id: str) -> Path | None:
    """Find the latest timestamped autointerp run directory."""
    autointerp_dir = get_autointerp_dir(run_id)
    if not autointerp_dir.exists():
        return None
    candidates: list[Path] = []
    for subdir in autointerp_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name in ("eval", "scoring"):
            continue
        scoring_dir = subdir / "scoring"
        if scoring_dir.exists():
            candidates.append(subdir)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def load_detection_scores(run_id: str) -> dict[str, float] | None:
    """Load detection eval scores from the latest autointerp run's scoring dir."""
    run_dir = _find_latest_autointerp_run_dir(run_id)
    if run_dir is None:
        return None
    path = _find_latest_jsonl(run_dir / "scoring" / "detection")
    if path is None:
        return None
    return _load_scores_from_jsonl(path)


def load_fuzzing_scores(run_id: str) -> dict[str, float] | None:
    """Load fuzzing eval scores from the latest autointerp run's scoring dir."""
    run_dir = _find_latest_autointerp_run_dir(run_id)
    if run_dir is None:
        return None
    path = _find_latest_jsonl(run_dir / "scoring" / "fuzzing")
    if path is None:
        return None
    return _load_scores_from_jsonl(path)
