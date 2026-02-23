"""Dataset attribution endpoints.

Serves pre-computed component-to-component attribution strengths aggregated
over the full training dataset.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors
from spd.dataset_attributions.storage import DatasetAttributionEntry as StorageEntry
from spd.dataset_attributions.storage import DatasetAttributionStorage


class DatasetAttributionEntry(BaseModel):
    """A single entry in attribution results."""

    component_key: str
    layer: str
    component_idx: int
    value: float


class DatasetAttributionMetadata(BaseModel):
    """Metadata about dataset attributions availability."""

    available: bool
    n_batches_processed: int | None
    n_tokens_processed: int | None
    n_component_layer_keys: int | None
    vocab_size: int | None
    d_model: int | None
    ci_threshold: float | None


class ComponentAttributions(BaseModel):
    """All attribution data for a single component (sources and targets, positive and negative)."""

    positive_sources: list[DatasetAttributionEntry]
    negative_sources: list[DatasetAttributionEntry]
    positive_targets: list[DatasetAttributionEntry]
    negative_targets: list[DatasetAttributionEntry]


router = APIRouter(prefix="/api/dataset_attributions", tags=["dataset_attributions"])

NOT_AVAILABLE_MSG = (
    "Dataset attributions not available. Run: spd-attributions <wandb_path> --n_batches N"
)


def _to_concrete_key(canonical_layer: str, component_idx: int, loaded: DepLoadedRun) -> str:
    """Translate canonical layer + idx to concrete storage key.

    "embed" maps to the concrete embedding path (e.g. "wte") in storage.
    "output" is a pseudo-layer used as-is in storage.
    """
    if canonical_layer == "output":
        return f"output:{component_idx}"
    concrete = loaded.topology.canon_to_target(canonical_layer)
    return f"{concrete}:{component_idx}"


def _require_storage(loaded: DepLoadedRun) -> DatasetAttributionStorage:
    """Get storage or raise 404."""
    if loaded.attributions is None:
        raise HTTPException(status_code=404, detail=NOT_AVAILABLE_MSG)
    return loaded.attributions.get_attributions()


def _require_source(storage: DatasetAttributionStorage, component_key: str) -> None:
    """Validate component exists as a source or raise 404."""
    if not storage.has_source(component_key):
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found as source in attributions",
        )


def _require_target(storage: DatasetAttributionStorage, component_key: str) -> None:
    """Validate component exists as a target or raise 404."""
    if not storage.has_target(component_key):
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found as target in attributions",
        )


def _get_w_unembed(loaded: DepLoadedRun) -> Float[Tensor, "d_model vocab"]:
    """Get the unembedding matrix from the loaded model."""
    return loaded.topology.get_unembed_weight()


def _to_api_entries(
    loaded: DepLoadedRun, entries: list[StorageEntry]
) -> list[DatasetAttributionEntry]:
    """Convert storage entries to API response format with canonical keys."""

    def _canonicalize_layer(layer: str) -> str:
        if layer == "output":
            return layer
        return loaded.topology.target_to_canon(layer)

    return [
        DatasetAttributionEntry(
            component_key=f"{_canonicalize_layer(e.layer)}:{e.component_idx}",
            layer=_canonicalize_layer(e.layer),
            component_idx=e.component_idx,
            value=e.value,
        )
        for e in entries
    ]


@router.get("/metadata")
@log_errors
def get_attribution_metadata(loaded: DepLoadedRun) -> DatasetAttributionMetadata:
    """Get metadata about dataset attributions availability."""
    if loaded.attributions is None:
        return DatasetAttributionMetadata(
            available=False,
            n_batches_processed=None,
            n_tokens_processed=None,
            n_component_layer_keys=None,
            vocab_size=None,
            d_model=None,
            ci_threshold=None,
        )
    storage = loaded.attributions.get_attributions()
    return DatasetAttributionMetadata(
        available=True,
        n_batches_processed=storage.n_batches_processed,
        n_tokens_processed=storage.n_tokens_processed,
        n_component_layer_keys=storage.n_components,
        vocab_size=storage.vocab_size,
        d_model=storage.d_model,
        ci_threshold=storage.ci_threshold,
    )


@router.get("/{layer}/{component_idx}")
@log_errors
def get_component_attributions(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
) -> ComponentAttributions:
    """Get all attribution data for a component (sources and targets, positive and negative)."""
    storage = _require_storage(loaded)
    component_key = _to_concrete_key(layer, component_idx, loaded)

    # Component can be both a source and a target, so we need to check both
    is_source = storage.has_source(component_key)
    is_target = storage.has_target(component_key)

    if not is_source and not is_target:
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found in attributions",
        )

    w_unembed = _get_w_unembed(loaded) if is_source else None

    return ComponentAttributions(
        positive_sources=_to_api_entries(
            loaded, storage.get_top_sources(component_key, k, "positive")
        )
        if is_target
        else [],
        negative_sources=_to_api_entries(
            loaded, storage.get_top_sources(component_key, k, "negative")
        )
        if is_target
        else [],
        positive_targets=_to_api_entries(
            loaded,
            storage.get_top_targets(
                component_key,
                k,
                "positive",
                w_unembed=w_unembed,
                include_outputs=w_unembed is not None,
            ),
        )
        if is_source
        else [],
        negative_targets=_to_api_entries(
            loaded,
            storage.get_top_targets(
                component_key,
                k,
                "negative",
                w_unembed=w_unembed,
                include_outputs=w_unembed is not None,
            ),
        )
        if is_source
        else [],
    )


@router.get("/{layer}/{component_idx}/sources")
@log_errors
def get_attribution_sources(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
    sign: Literal["positive", "negative"] = "positive",
) -> list[DatasetAttributionEntry]:
    """Get top-k source components that attribute TO this target over the dataset."""
    storage = _require_storage(loaded)
    target_key = _to_concrete_key(layer, component_idx, loaded)
    _require_target(storage, target_key)

    w_unembed = _get_w_unembed(loaded) if layer == "output" else None

    return _to_api_entries(
        loaded, storage.get_top_sources(target_key, k, sign, w_unembed=w_unembed)
    )


@router.get("/{layer}/{component_idx}/targets")
@log_errors
def get_attribution_targets(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
    sign: Literal["positive", "negative"] = "positive",
) -> list[DatasetAttributionEntry]:
    """Get top-k target components this source attributes TO over the dataset."""
    storage = _require_storage(loaded)
    source_key = _to_concrete_key(layer, component_idx, loaded)
    _require_source(storage, source_key)

    w_unembed = _get_w_unembed(loaded)

    return _to_api_entries(
        loaded, storage.get_top_targets(source_key, k, sign, w_unembed=w_unembed)
    )


@router.get("/between/{source_layer}/{source_idx}/{target_layer}/{target_idx}")
@log_errors
def get_attribution_between(
    source_layer: str,
    source_idx: int,
    target_layer: str,
    target_idx: int,
    loaded: DepLoadedRun,
) -> float:
    """Get attribution strength from source component to target component."""
    storage = _require_storage(loaded)
    source_key = _to_concrete_key(source_layer, source_idx, loaded)
    target_key = _to_concrete_key(target_layer, target_idx, loaded)
    _require_source(storage, source_key)
    _require_target(storage, target_key)

    w_unembed = _get_w_unembed(loaded) if target_layer == "output" else None

    return storage.get_attribution(source_key, target_key, w_unembed=w_unembed)
