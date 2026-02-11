"""Loaders for reading harvest output files.

Functions here are used by autointerp, dataset_attributions, and other modules
that import from spd.harvest.loaders directly (not via HarvestRepo).
"""

from spd.harvest.db import HarvestDB
from spd.harvest.schemas import (
    ComponentData,
    ComponentSummary,
    get_activation_contexts_dir,
    get_correlations_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage


def _open_harvest_db(wandb_run_id: str) -> HarvestDB:
    db_path = get_activation_contexts_dir(wandb_run_id) / "harvest.db"
    assert db_path.exists(), f"No harvest.db at {db_path}"
    return HarvestDB(db_path)


def load_harvest_ci_threshold(wandb_run_id: str) -> float:
    """Load the CI threshold used during harvest for this run."""
    db = _open_harvest_db(wandb_run_id)
    return db.get_ci_threshold()


def load_all_components(wandb_run_id: str) -> list[ComponentData]:
    """Load all components that fired during harvest.

    Reads the harvest ci_threshold from the DB and excludes components
    whose mean_ci is below it (i.e. components that effectively never fire).
    """
    db = _open_harvest_db(wandb_run_id)
    ci_threshold = db.get_ci_threshold()
    return db.get_all_components(ci_threshold)


def load_activation_contexts_summary(wandb_run_id: str) -> dict[str, ComponentSummary] | None:
    """Load lightweight summary of activation contexts (just metadata, not full examples)."""
    db_path = get_activation_contexts_dir(wandb_run_id) / "harvest.db"
    if not db_path.exists():
        return None
    db = HarvestDB(db_path)
    return db.get_summary()


def load_component_activation_contexts(wandb_run_id: str, component_key: str) -> ComponentData:
    """Load a single component's activation contexts."""
    db = _open_harvest_db(wandb_run_id)
    comp = db.get_component(component_key)
    assert comp is not None, f"Component {component_key} not found in harvest.db"
    return comp


def load_correlations(wandb_run_id: str) -> CorrelationStorage:
    """Load component correlations from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "component_correlations.pt"
    assert path.exists()
    return CorrelationStorage.load(path)


def load_token_stats(wandb_run_id: str) -> TokenStatsStorage:
    """Load token statistics from harvest output."""
    corr_dir = get_correlations_dir(wandb_run_id)
    path = corr_dir / "token_stats.pt"
    assert path.exists()
    return TokenStatsStorage.load(path)
