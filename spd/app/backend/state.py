"""Application state management for the SPD backend.

Contains:
- RunState: Runtime state for a loaded run (model, tokenizer, caches)
- StateManager: Singleton managing app-wide state with proper lifecycle
"""

from dataclasses import dataclass, field
from typing import Any

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.database import PromptAttrDB, Run
from spd.autointerp.loaders import (
    load_detection_scores,
    load_fuzzing_scores,
    load_interpretations,
    load_intruder_scores,
)
from spd.autointerp.schemas import InterpretationResult
from spd.configs import Config
from spd.dataset_attributions import DatasetAttributionStorage, load_dataset_attributions
from spd.harvest.loaders import (
    load_activation_contexts_summary,
    load_correlations,
    load_token_stats,
)
from spd.harvest.schemas import ComponentSummary
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.models.component_model import ComponentModel
from spd.topology import TransformerTopology

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
        self._dataset_attributions = _NOT_LOADED
        self._intruder_scores = _NOT_LOADED
        self._detection_scores = _NOT_LOADED
        self._fuzzing_scores = _NOT_LOADED

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
    def interpretations(self) -> dict[str, InterpretationResult] | None:
        if self._interpretations is _NOT_LOADED:
            self._interpretations = load_interpretations(self.run_id)
        if self._interpretations is None:
            return None
        assert isinstance(self._interpretations, dict)
        return self._interpretations

    def _load_activation_contexts_summary(self) -> dict[str, ComponentSummary] | None:
        if self._activation_contexts_summary is _NOT_LOADED:
            self._activation_contexts_summary = load_activation_contexts_summary(self.run_id)
        if self._activation_contexts_summary is None:
            return None
        assert isinstance(self._activation_contexts_summary, dict)
        return self._activation_contexts_summary

    def has_activation_contexts_summary(self) -> bool:
        """Check if activation contexts summary is available."""
        return self._load_activation_contexts_summary() is not None

    @property
    def activation_contexts_summary(self) -> dict[str, ComponentSummary]:
        """Lightweight summary of activation contexts, keyed by component_key (e.g. 'h.0.mlp.c_fc:5')."""
        result = self._load_activation_contexts_summary()
        assert result is not None, f"No activation contexts summary found for run {self.run_id}"
        return result

    def _load_dataset_attributions(self) -> DatasetAttributionStorage | None:
        if self._dataset_attributions is _NOT_LOADED:
            self._dataset_attributions = load_dataset_attributions(self.run_id)
        if self._dataset_attributions is None:
            return None
        assert isinstance(self._dataset_attributions, DatasetAttributionStorage)
        return self._dataset_attributions

    def has_dataset_attributions(self) -> bool:
        """Check if dataset attributions are available."""
        return self._load_dataset_attributions() is not None

    @property
    def dataset_attributions(self) -> DatasetAttributionStorage:
        """Dataset-aggregated attribution matrix."""
        result = self._load_dataset_attributions()
        assert result is not None, (
            f"No dataset attributions found for run {self.run_id}. "
            "Run: spd-attributions <wandb_path> --n_batches N"
        )
        return result

    @property
    def intruder_scores(self) -> dict[str, float] | None:
        if self._intruder_scores is _NOT_LOADED:
            self._intruder_scores = load_intruder_scores(self.run_id)
        if self._intruder_scores is None:
            return None
        assert isinstance(self._intruder_scores, dict)
        return self._intruder_scores

    @property
    def detection_scores(self) -> dict[str, float] | None:
        if self._detection_scores is _NOT_LOADED:
            self._detection_scores = load_detection_scores(self.run_id)
        if self._detection_scores is None:
            return None
        assert isinstance(self._detection_scores, dict)
        return self._detection_scores

    @property
    def fuzzing_scores(self) -> dict[str, float] | None:
        if self._fuzzing_scores is _NOT_LOADED:
            self._fuzzing_scores = load_fuzzing_scores(self.run_id)
        if self._fuzzing_scores is None:
            return None
        assert isinstance(self._fuzzing_scores, dict)
        return self._fuzzing_scores


@dataclass
class RunState:
    """Runtime state for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    topology: TransformerTopology
    tokenizer: AppTokenizer
    sources_by_target: dict[str, list[str]]
    config: Config
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

    db: PromptAttrDB
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

    def initialize(self, db: PromptAttrDB) -> None:
        """Initialize state with database connection."""
        self._state = AppState(db=db)

    @property
    def state(self) -> AppState:
        """Get app state. Fails fast if not initialized."""
        assert self._state is not None, "App state not initialized - lifespan not started"
        return self._state

    @property
    def db(self) -> PromptAttrDB:
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
