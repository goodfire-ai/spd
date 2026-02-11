"""Harvest data repository.

Owns SPD_OUT_DIR/harvest/<run_id>/ and provides read access to all harvest artifacts.
No in-memory caching -- reads go through on every call. Component data backed by SQLite;
correlations and token stats remain as .pt files.

Supports two layouts:
- Sub-run layout (current): harvest/<run_id>/h-YYYYMMDD_HHMMSS/{harvest.db, *.pt}
- Legacy layout (fallback): harvest/<run_id>/activation_contexts/harvest.db
                             harvest/<run_id>/correlations/{*.pt}
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
    """Read-only access to harvest data for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._db: HarvestDB | None = None

    def _find_latest_subrun(self) -> Path | None:
        """Find the latest sub-run directory, or fall back to legacy layout."""
        harvest_dir = get_harvest_dir(self.run_id)
        if not harvest_dir.exists():
            return None
        candidates = sorted(
            [d for d in harvest_dir.iterdir() if d.is_dir() and d.name.startswith("h-")],
            key=lambda d: d.name,
        )
        if candidates:
            return candidates[-1]
        # Legacy fallback: check for old activation_contexts/harvest.db layout
        if (harvest_dir / "activation_contexts" / "harvest.db").exists():
            return None
        return None

    def _resolve_db_path(self) -> Path | None:
        """Resolve the path to harvest.db, checking sub-run dirs then legacy layout."""
        subrun = self._find_latest_subrun()
        if subrun is not None:
            path = subrun / "harvest.db"
            return path if path.exists() else None
        # Legacy fallback
        legacy = get_harvest_dir(self.run_id) / "activation_contexts" / "harvest.db"
        return legacy if legacy.exists() else None

    def _resolve_data_dir(self) -> Path | None:
        """Resolve the directory containing correlations and token stats .pt files."""
        subrun = self._find_latest_subrun()
        if subrun is not None:
            return subrun
        # Legacy fallback
        legacy = get_harvest_dir(self.run_id) / "correlations"
        return legacy if legacy.exists() else None

    def _get_db(self) -> HarvestDB | None:
        """Lazily open the SQLite database on first access."""
        if self._db is not None:
            return self._db
        db_path = self._resolve_db_path()
        if db_path is None:
            return None
        self._db = HarvestDB(db_path)
        return self._db

    # -- Activation contexts ---------------------------------------------------

    def has_activation_contexts(self) -> bool:
        db = self._get_db()
        return db is not None and db.has_data()

    def get_summary(self) -> dict[str, ComponentSummary] | None:
        db = self._get_db()
        if db is None:
            return None
        return db.get_summary()

    def get_component(self, component_key: str) -> ComponentData | None:
        db = self._get_db()
        if db is None:
            return None
        return db.get_component(component_key)

    def get_components_bulk(self, component_keys: list[str]) -> dict[str, ComponentData]:
        db = self._get_db()
        if db is None:
            return {}
        return db.get_components_bulk(component_keys)

    def get_ci_threshold(self) -> float:
        db = self._get_db()
        assert db is not None, f"No harvest.db for run {self.run_id}"
        return db.get_ci_threshold()

    def get_all_components(self) -> list[ComponentData]:
        """Load all components with mean_ci above the harvest ci_threshold."""
        db = self._get_db()
        assert db is not None, f"No harvest.db for run {self.run_id}"
        return db.get_all_components(db.get_ci_threshold())

    # -- Correlations & token stats (tensor data) ------------------------------

    def has_correlations(self) -> bool:
        data_dir = self._resolve_data_dir()
        return data_dir is not None and (data_dir / "component_correlations.pt").exists()

    def get_correlations(self) -> CorrelationStorage | None:
        data_dir = self._resolve_data_dir()
        if data_dir is None:
            return None
        path = data_dir / "component_correlations.pt"
        if not path.exists():
            return None
        return CorrelationStorage.load(path)

    def has_token_stats(self) -> bool:
        data_dir = self._resolve_data_dir()
        return data_dir is not None and (data_dir / "token_stats.pt").exists()

    def get_token_stats(self) -> TokenStatsStorage | None:
        data_dir = self._resolve_data_dir()
        if data_dir is None:
            return None
        path = data_dir / "token_stats.pt"
        if not path.exists():
            return None
        return TokenStatsStorage.load(path)
