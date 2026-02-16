"""Harvest data repository.

Owns SPD_OUT_DIR/harvest/<run_id>/ and provides read access to all harvest artifacts.
No in-memory caching -- reads go through on every call. Component data backed by SQLite;
correlations and token stats remain as .pt files.

Use HarvestRepo.open() to construct — returns None if no harvest data exists.
Layout: harvest/<run_id>/h-YYYYMMDD_HHMMSS/{harvest.db, *.pt}
"""

from pathlib import Path

from spd.harvest.db import HarvestDB
from spd.harvest.schemas import (
    ComponentData,
    ComponentSummary,
    get_harvest_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage


class HarvestRepo:
    """Read-only access to harvest data for a single run.

    Constructed via HarvestRepo.open(). DB is opened eagerly at construction.
    """

    def __init__(self, db: HarvestDB, subrun_dir: Path, run_id: str) -> None:
        self._db = db
        self._subrun_dir = subrun_dir
        self.db_path = subrun_dir / "harvest.db"
        self.subrun_id = subrun_dir.name
        self.run_id = run_id

    @classmethod
    def open(cls, run_id: str, subrun_id: str | None = None) -> "HarvestRepo | None":
        """Open harvest data for a run. Returns None if no harvest data exists."""
        harvest_dir = get_harvest_dir(run_id)
        if not harvest_dir.exists():
            return None

        if subrun_id is not None:
            subrun_dir = harvest_dir / subrun_id
        else:
            candidates = sorted(
                [d for d in harvest_dir.iterdir() if d.is_dir() and d.name.startswith("h-")],
                key=lambda d: d.name,
            )
            if not candidates:
                return None
            subrun_dir = candidates[-1]

        db_path = subrun_dir / "harvest.db"
        if not db_path.exists():
            return None

        return cls(
            db=HarvestDB(db_path, readonly=True),
            subrun_dir=subrun_dir,
            run_id=run_id,
        )

    # -- Provenance ------------------------------------------------------------

    def get_config(self) -> dict[str, object]:
        return self._db.get_config_dict()

    def get_component_count(self) -> int:
        return self._db.get_component_count()

    # -- Activation contexts ---------------------------------------------------

    def get_summary(self) -> dict[str, ComponentSummary]:
        return self._db.get_summary()

    def get_component(self, component_key: str) -> ComponentData | None:
        return self._db.get_component(component_key)

    def get_components_bulk(self, component_keys: list[str]) -> dict[str, ComponentData]:
        return self._db.get_components_bulk(component_keys)

    def get_activation_threshold(self) -> float:
        return self._db.get_activation_threshold()

    def get_all_components(self) -> list[ComponentData]:
        return self._db.get_all_components(self._db.get_activation_threshold())

    # -- Correlations & token stats (tensor data) ------------------------------

    def get_correlations(self) -> CorrelationStorage | None:
        path = self._subrun_dir / "component_correlations.pt"
        if not path.exists():
            return None
        return CorrelationStorage.load(path)

    def get_token_stats(self) -> TokenStatsStorage | None:
        path = self._subrun_dir / "token_stats.pt"
        if not path.exists():
            return None
        return TokenStatsStorage.load(path)

    # -- Eval scores (e.g. intruder) -------------------------------------------

    def get_intruder_scores(self) -> dict[str, float] | None:
        scores = self._db.get_scores("intruder")
        return scores if scores else None
