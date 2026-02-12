"""Autointerp data repository.

Owns SPD_OUT_DIR/autointerp/<run_id>/ and provides read/write access to
interpretations and evaluation scores. Backed by SQLite (interp.db).
"""

from spd.autointerp.db import InterpDB
from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


class InterpRepo:
    """Read/write access to autointerp data for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._db: InterpDB | None = None

    def _get_db(self) -> InterpDB | None:
        if self._db is not None:
            return self._db
        db_path = get_autointerp_dir(self.run_id) / "interp.db"
        if not db_path.exists():
            return None
        self._db = InterpDB(db_path, readonly=True)
        return self._db

    def _get_or_create_db(self) -> InterpDB:
        if self._db is not None:
            return self._db
        autointerp_dir = get_autointerp_dir(self.run_id)
        autointerp_dir.mkdir(parents=True, exist_ok=True)
        db_path = autointerp_dir / "interp.db"
        self._db = InterpDB(db_path)
        return self._db

    # -- Interpretations -------------------------------------------------------

    def has_interpretations(self) -> bool:
        db = self._get_db()
        return db is not None and db.has_interpretations()

    def get_all_interpretations(self) -> dict[str, InterpretationResult] | None:
        db = self._get_db()
        if db is None:
            return None
        result = db.get_all_interpretations()
        return result if result else None

    def get_interpretation(self, component_key: str) -> InterpretationResult | None:
        db = self._get_db()
        if db is None:
            return None
        return db.get_interpretation(component_key)

    def save_interpretation(self, result: InterpretationResult) -> None:
        db = self._get_or_create_db()
        db.save_interpretation(result)

    # -- Eval scores (label-dependent only) ------------------------------------

    def get_detection_scores(self) -> dict[str, float] | None:
        db = self._get_db()
        if db is None:
            return None
        scores = db.get_scores("detection")
        return scores if scores else None

    def get_fuzzing_scores(self) -> dict[str, float] | None:
        db = self._get_db()
        if db is None:
            return None
        scores = db.get_scores("fuzzing")
        return scores if scores else None
