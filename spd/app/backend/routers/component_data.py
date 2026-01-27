"""Bundled component data endpoint.

Returns all ComponentNodeCard data in a single response to eliminate
HTTP roundtrip overhead (significant when using SSH port forwarding).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.routers.activation_contexts import (
    BulkActivationContextsRequest,
    get_activation_context_detail,
    get_activation_contexts_bulk,
)
from spd.app.backend.routers.correlations import (
    BulkCorrelationsRequest,
    BulkTokenStatsRequest,
    ComponentCorrelationsResponse,
    InterpretationDetail,
    TokenStatsResponse,
    get_component_correlations,
    get_component_correlations_bulk,
    get_component_token_stats,
    get_component_token_stats_bulk,
    get_interpretation_detail,
)
from spd.app.backend.routers.dataset_attributions import (
    ComponentAttributions,
    get_component_attributions,
)
from spd.app.backend.schemas import SubcomponentActivationContexts
from spd.app.backend.utils import log_errors, log_timing

router = APIRouter(prefix="/api/component_data", tags=["component_data"])


class ComponentDataBundle(BaseModel):
    """All ComponentNodeCard data in a single response."""

    component_detail: SubcomponentActivationContexts | None
    correlations: ComponentCorrelationsResponse | None
    token_stats: TokenStatsResponse | None
    attributions: ComponentAttributions | None
    interpretation_detail: InterpretationDetail | None
    errors: dict[str, str]


@router.get("/{layer}/{component_idx}")
@log_timing
@log_errors
def get_component_data_bundle(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k_corr: int = 100,
    top_k_tokens: int = 50,
    top_k_attrs: int = 20,
    detail_limit: int = 30,
) -> ComponentDataBundle:
    """Bundled endpoint for ComponentNodeCard data.

    Returns all component data in a single response. Missing data sources
    return None with error details in 'errors' dict.

    This eliminates 4 HTTP roundtrips vs the 5 individual endpoints,
    saving ~400ms+ over SSH tunnels.
    """
    errors: dict[str, str] = {}

    # Fetch component detail (activation contexts)
    component_detail: SubcomponentActivationContexts | None = None
    try:
        component_detail = get_activation_context_detail(layer, component_idx, loaded, detail_limit)
    except HTTPException as e:
        errors["component_detail"] = str(e.detail)
    except Exception as e:
        errors["component_detail"] = str(e)

    # Fetch correlations
    correlations: ComponentCorrelationsResponse | None = None
    try:
        correlations = get_component_correlations(layer, component_idx, loaded, top_k_corr)
    except HTTPException as e:
        errors["correlations"] = str(e.detail)
    except Exception as e:
        errors["correlations"] = str(e)

    # Fetch token stats
    token_stats: TokenStatsResponse | None = None
    try:
        token_stats = get_component_token_stats(layer, component_idx, loaded, top_k_tokens)
    except HTTPException as e:
        errors["token_stats"] = str(e.detail)
    except Exception as e:
        errors["token_stats"] = str(e)

    # Fetch dataset attributions
    attributions: ComponentAttributions | None = None
    try:
        attributions = get_component_attributions(layer, component_idx, loaded, top_k_attrs)
    except HTTPException as e:
        errors["attributions"] = str(e.detail)
    except Exception as e:
        errors["attributions"] = str(e)

    # Fetch interpretation detail
    interpretation_detail: InterpretationDetail | None = None
    try:
        interpretation_detail = get_interpretation_detail(layer, component_idx, loaded)
    except HTTPException as e:
        errors["interpretation_detail"] = str(e.detail)
    except Exception as e:
        errors["interpretation_detail"] = str(e)

    return ComponentDataBundle(
        component_detail=component_detail,
        correlations=correlations,
        token_stats=token_stats,
        attributions=attributions,
        interpretation_detail=interpretation_detail,
        errors=errors,
    )


class BulkComponentDataRequest(BaseModel):
    """Request for bulk component data fetch."""

    component_keys: list[str]  # ["h.0.mlp.c_fc:5", "h.1.attn.q_proj:12", ...]
    activation_contexts_limit: int = 30
    correlations_top_k: int = 20
    token_stats_top_k: int = 30


class BulkComponentDataResponse(BaseModel):
    """Combined bulk response for all component data types."""

    activation_contexts: dict[str, SubcomponentActivationContexts]
    correlations: dict[str, ComponentCorrelationsResponse]
    token_stats: dict[str, TokenStatsResponse]


@router.post("/bulk")
@log_timing
@log_errors
def get_component_data_bulk(
    request: BulkComponentDataRequest,
    loaded: DepLoadedRun,
) -> BulkComponentDataResponse:
    """Bulk fetch all component data in a single request.

    Combines activation contexts, correlations, and token stats into one response.
    This eliminates GIL contention from multiple concurrent requests and reduces
    HTTP roundtrips from 3 to 1.
    """
    # Fetch all data sequentially (no GIL contention)
    activation_contexts = get_activation_contexts_bulk(
        BulkActivationContextsRequest(
            component_keys=request.component_keys,
            limit=request.activation_contexts_limit,
        ),
        loaded,
    )

    correlations = get_component_correlations_bulk(
        BulkCorrelationsRequest(
            component_keys=request.component_keys,
            top_k=request.correlations_top_k,
        ),
        loaded,
    )

    token_stats = get_component_token_stats_bulk(
        BulkTokenStatsRequest(
            component_keys=request.component_keys,
            top_k=request.token_stats_top_k,
        ),
        loaded,
    )

    return BulkComponentDataResponse(
        activation_contexts=activation_contexts,
        correlations=correlations,
        token_stats=token_stats,
    )
