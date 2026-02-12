"""Loaders for reading autointerp output files."""

from spd.autointerp.db import InterpDB
from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


def _open_db(wandb_run_id: str) -> InterpDB | None:
    db_path = get_autointerp_dir(wandb_run_id) / "interp.db"
    if not db_path.exists():
        return None
    return InterpDB(db_path, readonly=True)


def load_interpretations(
    wandb_run_id: str,
    _autointerp_run_id: str | None = None,
) -> dict[str, InterpretationResult] | None:
    """Load interpretation results.

    Args:
        wandb_run_id: The SPD run ID.
        autointerp_run_id: Ignored (kept for API compatibility). All interpretations
            are stored in a single DB now.
    """
    db = _open_db(wandb_run_id)
    if db is None:
        return None
    result = db.get_all_interpretations()
    return result if result else None


def load_intruder_scores(run_id: str) -> dict[str, float] | None:
    db = _open_db(run_id)
    if db is None:
        return None
    scores = db.get_scores("intruder")
    return scores if scores else None


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
