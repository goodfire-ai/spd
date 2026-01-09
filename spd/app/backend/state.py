"""Application state management for the SPD backend.

Contains:
- RunState: Runtime state for a loaded run (model, tokenizer, caches)
- StateManager: Singleton managing app-wide state with proper lifecycle
"""

from dataclasses import dataclass, field
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.database import LocalAttrDB, Run
from spd.autointerp.loaders import load_interpretations
from spd.autointerp.schemas import InterpretationResult
from spd.configs import Config
from spd.harvest.loaders import (
    load_activation_contexts_summary,
    load_correlations,
    load_token_stats,
)
from spd.harvest.schemas import ComponentSummary
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.models.component_model import ComponentModel

_NOT_LOADED = object()


class HarvestCache:
    """Lazily-loaded harvest data for a run.

    All fields are loaded on first access and cached for the lifetime of the run.
    Uses a sentinel pattern to distinguish "not loaded" from "loaded but None".
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._correlations = _NOT_LOADED
        self._token_stats = _NOT_LOADED
        self._interpretations = _NOT_LOADED
        self._activation_contexts_summary = _NOT_LOADED

    @property
    def correlations(self) -> CorrelationStorage:
        if self._correlations is _NOT_LOADED:
            self._correlations = load_correlations(self.run_id)
        assert isinstance(self._correlations, CorrelationStorage)
        return self._correlations

    @property
    def token_stats(self) -> TokenStatsStorage:
        if self._token_stats is _NOT_LOADED:
            self._token_stats = load_token_stats(self.run_id)
        assert isinstance(self._token_stats, TokenStatsStorage)
        return self._token_stats

    @property
    def interpretations(self) -> dict[str, InterpretationResult]:
        if self._interpretations is _NOT_LOADED:
            self._interpretations = load_interpretations(self.run_id)
        assert isinstance(self._interpretations, dict)
        return self._interpretations

    @property
    def activation_contexts_summary(self) -> dict[str, ComponentSummary]:
        """Lightweight summary of activation contexts, keyed by component_key (e.g. 'h.0.mlp.c_fc:5')."""
        if self._activation_contexts_summary is _NOT_LOADED:
            self._activation_contexts_summary = load_activation_contexts_summary(self.run_id)
        assert isinstance(self._activation_contexts_summary, dict)
        return self._activation_contexts_summary


@dataclass
class RunState:
    """Runtime state for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    tokenizer: PreTrainedTokenizerBase
    sources_by_target: dict[str, list[str]]
    config: Config
    token_strings: dict[int, str]
    context_length: int
    harvest: HarvestCache


@dataclass
class DatasetSearchState:
    """State for dataset search results (memory-only, no persistence)."""

    results: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class AppState:
    """Server state. DB is always available; run_state is set after /api/runs/load."""

    db: LocalAttrDB
    run_state: RunState | None = field(default=None)
    dataset_search_state: DatasetSearchState | None = field(default=None)


class StateManager:
    """Singleton managing app state with proper lifecycle.

    Use StateManager.get() to access the singleton instance.
    The instance is initialized during FastAPI lifespan startup.
    """

    _instance: "StateManager | None" = None

    def __init__(self) -> None:
        self._state: AppState | None = None

    @classmethod
    def get(cls) -> "StateManager":
        """Get the singleton instance, creating if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def initialize(self, db: LocalAttrDB) -> None:
        """Initialize state with database connection."""
        self._state = AppState(db=db)

    @property
    def state(self) -> AppState:
        """Get app state. Fails fast if not initialized."""
        assert self._state is not None, "App state not initialized - lifespan not started"
        return self._state

    @property
    def db(self) -> LocalAttrDB:
        """Get database connection."""
        return self.state.db

    @property
    def run_state(self) -> RunState | None:
        """Get loaded run state (may be None)."""
        return self.state.run_state

    @run_state.setter
    def run_state(self, value: RunState | None) -> None:
        """Set loaded run state."""
        self.state.run_state = value

    def close(self) -> None:
        """Clean up resources."""
        if self._state is not None:
            self._state.db.close()
