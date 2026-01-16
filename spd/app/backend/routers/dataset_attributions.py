"""Dataset attribution endpoints.

Serves pre-computed component-to-component attribution strengths aggregated
over the full training dataset.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors


class DatasetAttributionEntry(BaseModel):
    """A single entry in attribution results."""

    component_key: str
    layer: str
    component_idx: int
    value: float


class DatasetAttributionMetadata(BaseModel):
    """Metadata about dataset attributions."""

    available: bool
    n_batches_processed: int | None
    n_tokens_processed: int | None
    n_components: int | None
    ci_threshold: float | None


router = APIRouter(prefix="/api/dataset_attributions", tags=["dataset_attributions"])


@router.get("/metadata")
@log_errors
def get_attribution_metadata(
    loaded: DepLoadedRun,
) -> DatasetAttributionMetadata:
    """Get metadata about dataset attributions availability."""
    storage = loaded.harvest.dataset_attributions
    if storage is None:
        return DatasetAttributionMetadata(
            available=False,
            n_batches_processed=None,
            n_tokens_processed=None,
            n_components=None,
            ci_threshold=None,
        )

    return DatasetAttributionMetadata(
        available=True,
        n_batches_processed=storage.n_batches_processed,
        n_tokens_processed=storage.n_tokens_processed,
        n_components=len(storage.component_keys),
        ci_threshold=storage.ci_threshold,
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
    """Get top-k components that attribute TO this component over the dataset.

    Returns components that most strongly influence the target component,
    aggregated across all dataset samples.
    """
    storage = loaded.harvest.dataset_attributions
    if storage is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset attributions not available. Run: python -m spd.dataset_attributions.scripts.run <wandb_path> --n_batches N",
        )

    component_key = f"{layer}:{component_idx}"
    if component_key not in storage._key_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found in attributions",
        )

    entries = storage.get_top_sources(component_key, k, sign)
    return [
        DatasetAttributionEntry(
            component_key=e.component_key,
            layer=e.layer,
            component_idx=e.component_idx,
            value=e.value,
        )
        for e in entries
    ]


@router.get("/{layer}/{component_idx}/targets")
@log_errors
def get_attribution_targets(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
    sign: Literal["positive", "negative"] = "positive",
) -> list[DatasetAttributionEntry]:
    """Get top-k components this component attributes TO over the dataset.

    Returns components that the source component most strongly influences,
    aggregated across all dataset samples.
    """
    storage = loaded.harvest.dataset_attributions
    if storage is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset attributions not available. Run: python -m spd.dataset_attributions.scripts.run <wandb_path> --n_batches N",
        )

    component_key = f"{layer}:{component_idx}"
    if component_key not in storage._key_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found in attributions",
        )

    entries = storage.get_top_targets(component_key, k, sign)
    return [
        DatasetAttributionEntry(
            component_key=e.component_key,
            layer=e.layer,
            component_idx=e.component_idx,
            value=e.value,
        )
        for e in entries
    ]


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
    storage = loaded.harvest.dataset_attributions
    if storage is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset attributions not available. Run: python -m spd.dataset_attributions.scripts.run <wandb_path> --n_batches N",
        )

    source_key = f"{source_layer}:{source_idx}"
    target_key = f"{target_layer}:{target_idx}"

    if source_key not in storage._key_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Source component {source_key} not found in attributions",
        )
    if target_key not in storage._key_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Target component {target_key} not found in attributions",
        )

    return storage.get_attribution(source_key, target_key)
