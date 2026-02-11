"""Harvest data repository.

Owns SPD_OUT_DIR/harvest/<run_id>/ and provides read access to all harvest artifacts.
No in-memory caching -- reads go through on every call. Component data backed by SQLite;
correlations and token stats remain as .pt files.
"""

from spd.harvest.db import HarvestDB
from spd.harvest.schemas import (
    ComponentData,
    ComponentSummary,
    get_activation_contexts_dir,
    get_correlations_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage


class HarvestRepo:
    """Read-only access to harvest data for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._ac_dir = get_activation_contexts_dir(run_id)
        self._corr_dir = get_correlations_dir(run_id)
        self._db: HarvestDB | None = None

    def _get_db(self) -> HarvestDB | None:
        """Lazily open the SQLite database on first access."""
        if self._db is not None:
            return self._db
        db_path = self._ac_dir / "harvest.db"
        if not db_path.exists():
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
        return (self._corr_dir / "component_correlations.pt").exists()

    def get_correlations(self) -> CorrelationStorage | None:
        path = self._corr_dir / "component_correlations.pt"
        if not path.exists():
            return None
        return CorrelationStorage.load(path)

    def has_token_stats(self) -> bool:
        return (self._corr_dir / "token_stats.pt").exists()

    def get_token_stats(self) -> TokenStatsStorage | None:
        path = self._corr_dir / "token_stats.pt"
        if not path.exists():
            return None
        return TokenStatsStorage.load(path)
