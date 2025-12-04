"""Application state management for the SPD backend.

Contains:
- RunState: Runtime state for a loaded run (model, tokenizer, caches)
- StateManager: Singleton managing app-wide state with proper lifecycle
"""

from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.db import LocalAttrDB
from spd.app.backend.db.database import Run
from spd.app.backend.schemas import ModelActivationContexts
from spd.configs import Config
from spd.models.component_model import ComponentModel


@dataclass
class RunState:
    """Runtime state for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    tokenizer: PreTrainedTokenizerBase
    sources_by_target: dict[str, list[str]]
    config: Config
    token_strings: dict[int, str]
    train_loader: DataLoader[Any]
    activation_contexts_cache: ModelActivationContexts | None = None


@dataclass
class AppState:
    """Server state. DB is always available; run_state is set after /api/runs/load."""

    db: LocalAttrDB
    run_state: RunState | None = field(default=None)


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
