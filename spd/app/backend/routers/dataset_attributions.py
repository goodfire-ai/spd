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
from spd.dataset_attributions.storage import AttrMetric, DatasetAttributionStorage
from spd.dataset_attributions.storage import DatasetAttributionEntry as StorageEntry

ATTR_METRICS: list[AttrMetric] = ["attr", "attr_abs", "mean_squared_attr"]


class DatasetAttributionEntry(BaseModel):
    component_key: str
    layer: str
    component_idx: int
    value: float
    token_str: str | None = None


class DatasetAttributionMetadata(BaseModel):
    available: bool
    n_batches_processed: int | None
    n_tokens_processed: int | None
    n_component_layer_keys: int | None
    ci_threshold: float | None


class ComponentAttributions(BaseModel):
    positive_sources: list[DatasetAttributionEntry]
    negative_sources: list[DatasetAttributionEntry]
    positive_targets: list[DatasetAttributionEntry]
    negative_targets: list[DatasetAttributionEntry]


class AllMetricAttributions(BaseModel):
    attr: ComponentAttributions
    attr_abs: ComponentAttributions
    mean_squared_attr: ComponentAttributions


router = APIRouter(prefix="/api/dataset_attributions", tags=["dataset_attributions"])

NOT_AVAILABLE_MSG = (
    "Dataset attributions not available. Run: spd-attributions <wandb_path> --n_batches N"
)


def _storage_key(canonical_layer: str, component_idx: int) -> str:
    return f"{canonical_layer}:{component_idx}"


def _require_storage(loaded: DepLoadedRun) -> DatasetAttributionStorage:
    if loaded.attributions is None:
        raise HTTPException(status_code=404, detail=NOT_AVAILABLE_MSG)
    return loaded.attributions.get_attributions()


def _require_source(storage: DatasetAttributionStorage, component_key: str) -> None:
    if not storage.has_source(component_key):
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found as source in attributions",
        )


def _require_target(storage: DatasetAttributionStorage, component_key: str) -> None:
    if not storage.has_target(component_key):
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found as target in attributions",
        )


def _get_w_unembed(loaded: DepLoadedRun) -> Float[Tensor, "d_model vocab"]:
    return loaded.topology.get_unembed_weight()


def _to_api_entries(
    entries: list[StorageEntry], loaded: DepLoadedRun
) -> list[DatasetAttributionEntry]:
    return [
        DatasetAttributionEntry(
            component_key=e.component_key,
            layer=e.layer,
            component_idx=e.component_idx,
            value=e.value,
            token_str=loaded.tokenizer.decode([e.component_idx])
            if e.layer in ("embed", "output")
            else None,
        )
        for e in entries
    ]


def _get_component_attributions_for_metric(
    storage: DatasetAttributionStorage,
    loaded: DepLoadedRun,
    component_key: str,
    k: int,
    metric: AttrMetric,
    is_source: bool,
    is_target: bool,
    w_unembed: Float[Tensor, "d_model vocab"] | None,
) -> ComponentAttributions:
    return ComponentAttributions(
        positive_sources=_to_api_entries(
            storage.get_top_sources(component_key, k, "positive", metric), loaded
        )
        if is_target
        else [],
        negative_sources=_to_api_entries(
            storage.get_top_sources(component_key, k, "negative", metric), loaded
        )
        if is_target
        else [],
        positive_targets=_to_api_entries(
            storage.get_top_targets(
                component_key,
                k,
                "positive",
                metric,
                w_unembed=w_unembed,
                include_outputs=w_unembed is not None,
            ),
            loaded,
        )
        if is_source
        else [],
        negative_targets=_to_api_entries(
            storage.get_top_targets(
                component_key,
                k,
                "negative",
                metric,
                w_unembed=w_unembed,
                include_outputs=w_unembed is not None,
            ),
            loaded,
        )
        if is_source
        else [],
    )


@router.get("/metadata")
@log_errors
def get_attribution_metadata(loaded: DepLoadedRun) -> DatasetAttributionMetadata:
    if loaded.attributions is None:
        return DatasetAttributionMetadata(
            available=False,
            n_batches_processed=None,
            n_tokens_processed=None,
            n_component_layer_keys=None,
            ci_threshold=None,
        )
    storage = loaded.attributions.get_attributions()
    return DatasetAttributionMetadata(
        available=True,
        n_batches_processed=storage.n_batches_processed,
        n_tokens_processed=storage.n_tokens_processed,
        n_component_layer_keys=storage.n_components,
        ci_threshold=storage.ci_threshold,
    )


@router.get("/{layer}/{component_idx}")
@log_errors
def get_component_attributions(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
) -> AllMetricAttributions:
    """Get all attribution data for a component across all 3 metrics."""
    storage = _require_storage(loaded)
    component_key = _storage_key(layer, component_idx)

    is_source = storage.has_source(component_key)
    is_target = storage.has_target(component_key)

    if not is_source and not is_target:
        raise HTTPException(
            status_code=404,
            detail=f"Component {component_key} not found in attributions",
        )

    w_unembed = _get_w_unembed(loaded) if is_source else None

    return AllMetricAttributions(
        **{
            metric: _get_component_attributions_for_metric(
                storage, loaded, component_key, k, metric, is_source, is_target, w_unembed
            )
            for metric in ATTR_METRICS
        }
    )


@router.get("/{layer}/{component_idx}/sources")
@log_errors
def get_attribution_sources(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
    sign: Literal["positive", "negative"] = "positive",
    metric: AttrMetric = "attr",
) -> list[DatasetAttributionEntry]:
    storage = _require_storage(loaded)
    target_key = _storage_key(layer, component_idx)
    _require_target(storage, target_key)

    w_unembed = _get_w_unembed(loaded) if layer == "output" else None

    return _to_api_entries(
        storage.get_top_sources(target_key, k, sign, metric, w_unembed=w_unembed), loaded
    )


@router.get("/{layer}/{component_idx}/targets")
@log_errors
def get_attribution_targets(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    k: Annotated[int, Query(ge=1)] = 10,
    sign: Literal["positive", "negative"] = "positive",
    metric: AttrMetric = "attr",
) -> list[DatasetAttributionEntry]:
    storage = _require_storage(loaded)
    source_key = _storage_key(layer, component_idx)
    _require_source(storage, source_key)

    w_unembed = _get_w_unembed(loaded)

    return _to_api_entries(
        storage.get_top_targets(source_key, k, sign, metric, w_unembed=w_unembed), loaded
    )
