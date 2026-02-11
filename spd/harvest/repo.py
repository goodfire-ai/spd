"""Harvest data repository.

Owns SPD_OUT_DIR/harvest/<run_id>/ and provides read access to all harvest artifacts.
No in-memory caching — reads go through on every call. Currently backed by files
(JSONL, JSON, .pt); will migrate component-level data to SQLite.
"""

from pathlib import Path

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

    # ── Activation contexts ───────────────────────────────────────────

    def has_activation_contexts(self) -> bool:
        return (self._ac_dir / "summary.json").exists()

    def get_summary(self) -> dict[str, ComponentSummary] | None:
        path = self._ac_dir / "summary.json"
        if not path.exists():
            return None
        return ComponentSummary.load_all(path)

    def get_component(self, component_key: str) -> ComponentData | None:
        """Load a single component's full data (examples, PMI, etc.)."""
        from spd.harvest.loaders import load_component_activation_contexts

        if not (self._ac_dir / "components.jsonl").exists():
            return None
        return load_component_activation_contexts(self.run_id, component_key)

    def get_components_bulk(self, component_keys: list[str]) -> dict[str, ComponentData]:
        """Load multiple components in a single pass."""
        from spd.harvest.loaders import load_component_activation_contexts_bulk

        if not (self._ac_dir / "components.jsonl").exists():
            return {}
        return load_component_activation_contexts_bulk(self.run_id, component_keys)

    def get_ci_threshold(self) -> float:
        from spd.harvest.loaders import load_harvest_ci_threshold

        return load_harvest_ci_threshold(self.run_id)

    # ── Correlations & token stats (tensor data) ──────────────────────

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
