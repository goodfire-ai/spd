"""Autointerp data repository.

Owns SPD_OUT_DIR/autointerp/<run_id>/ and provides read/write access to
interpretations and evaluation scores. No in-memory caching.
"""

from spd.autointerp.loaders import (
    load_detection_scores,
    load_fuzzing_scores,
    load_interpretations,
    load_intruder_scores,
)
from spd.autointerp.schemas import InterpretationResult


class InterpRepo:
    """Read access to autointerp data for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    # ── Interpretations ───────────────────────────────────────────────

    def get_all_interpretations(self) -> dict[str, InterpretationResult] | None:
        return load_interpretations(self.run_id)

    def get_interpretation(self, component_key: str) -> InterpretationResult | None:
        interps = self.get_all_interpretations()
        if interps is None:
            return None
        return interps.get(component_key)

    # ── Eval scores ───────────────────────────────────────────────────

    def get_intruder_scores(self) -> dict[str, float] | None:
        return load_intruder_scores(self.run_id)

    def get_detection_scores(self) -> dict[str, float] | None:
        return load_detection_scores(self.run_id)

    def get_fuzzing_scores(self) -> dict[str, float] | None:
        return load_fuzzing_scores(self.run_id)
