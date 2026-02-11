"""Bulk component data endpoint for prefetching."""

from fastapi import APIRouter
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.routers.activation_contexts import (
    BulkActivationContextsRequest,
    get_activation_contexts_bulk,
)
from spd.app.backend.routers.correlations import (
    BulkCorrelationsRequest,
    BulkTokenStatsRequest,
    ComponentCorrelationsResponse,
    TokenStatsResponse,
    get_component_correlations_bulk,
    get_component_token_stats_bulk,
)
from spd.app.backend.schemas import SubcomponentActivationContexts
from spd.app.backend.utils import log_errors

router = APIRouter(prefix="/api/component_data", tags=["component_data"])


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
    # Each section returns empty dict if harvest data is unavailable
    try:
        activation_contexts = get_activation_contexts_bulk(
            BulkActivationContextsRequest(
                component_keys=request.component_keys,
                limit=request.activation_contexts_limit,
            ),
            loaded,
        )
    except (AssertionError, FileNotFoundError):
        activation_contexts = {}

    try:
        correlations = get_component_correlations_bulk(
            BulkCorrelationsRequest(
                component_keys=request.component_keys,
                top_k=request.correlations_top_k,
            ),
            loaded,
        )
    except (AssertionError, FileNotFoundError):
        correlations = {}

    try:
        token_stats = get_component_token_stats_bulk(
            BulkTokenStatsRequest(
                component_keys=request.component_keys,
                top_k=request.token_stats_top_k,
            ),
            loaded,
        )
    except (AssertionError, FileNotFoundError):
        token_stats = {}

    return BulkComponentDataResponse(
        activation_contexts=activation_contexts,
        correlations=correlations,
        token_stats=token_stats,
    )
