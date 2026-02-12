"""Loaders for reading autointerp output files."""

from pathlib import Path

from spd.autointerp.db import InterpDB
from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


def _find_latest_subrun_dir(wandb_run_id: str) -> Path | None:
    autointerp_dir = get_autointerp_dir(wandb_run_id)
    if not autointerp_dir.exists():
        return None
    candidates = sorted(
        [d for d in autointerp_dir.iterdir() if d.is_dir() and d.name.startswith("a-")],
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def _open_db(wandb_run_id: str) -> InterpDB | None:
    subrun_dir = _find_latest_subrun_dir(wandb_run_id)
    if subrun_dir is None:
        return None
    db_path = subrun_dir / "interp.db"
    if not db_path.exists():
        return None
    return InterpDB(db_path, readonly=True)


def load_interpretations(wandb_run_id: str) -> dict[str, InterpretationResult] | None:
    db = _open_db(wandb_run_id)
    if db is None:
        return None
    result = db.get_all_interpretations()
    return result if result else None


def load_detection_scores(run_id: str) -> dict[str, float] | None:
    db = _open_db(run_id)
    if db is None:
        return None
    scores = db.get_scores("detection")
    return scores if scores else None


def load_fuzzing_scores(run_id: str) -> dict[str, float] | None:
    db = _open_db(run_id)
    if db is None:
        return None
    scores = db.get_scores("fuzzing")
    return scores if scores else None
