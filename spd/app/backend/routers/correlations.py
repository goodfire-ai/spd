"""Component correlation and interpretation endpoints.

These endpoints serve data produced by the harvest pipeline (spd.harvest),
which computes component co-occurrence statistics, token associations, and interpretations.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors
from spd.harvest import analysis
from spd.log import logger
from spd.topology import CanonicalWeight, TransformerTopology


def _canonical_to_concrete_key(
    canonical_layer: str, component_idx: int, topology: TransformerTopology
) -> str:
    """Translate canonical layer address + component idx to concrete component key for harvest data."""
    concrete = topology.get_target_module_path(CanonicalWeight.parse(canonical_layer))
    return f"{concrete}:{component_idx}"


def _concrete_to_canonical_key(concrete_key: str, topology: TransformerTopology) -> str:
    """Translate concrete component key to canonical component key."""
    layer, idx = concrete_key.rsplit(":", 1)
    canonical = topology.get_canonical_weight(layer).canonical_str()
    return f"{canonical}:{idx}"


class CorrelatedComponent(BaseModel):
    """A component correlated with a query component."""

    component_key: str
    score: float
    count_i: int  # Subject (query component) firing count
    count_j: int  # Object (this component) firing count
    count_ij: int  # Co-occurrence count
    n_tokens: int  # Total tokens


class ComponentCorrelationsResponse(BaseModel):
    """Correlation data for a component across different metrics."""

    precision: list[CorrelatedComponent]
    recall: list[CorrelatedComponent]
    jaccard: list[CorrelatedComponent]
    pmi: list[CorrelatedComponent]
    bottom_pmi: list[CorrelatedComponent]


class TokenPRLiftPMI(BaseModel):
    """Token precision, recall, lift, and PMI lists."""

    top_recall: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_precision: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_lift: list[tuple[str, float]]  # [(token, lift), ...] sorted desc
    top_pmi: list[tuple[str, float]]  # [(token, pmi), ...] highest positive association
    bottom_pmi: list[tuple[str, float]] | None  # [(token, pmi), ...] highest negative association


class TokenStatsResponse(BaseModel):
    """Token stats for a component (from batch job).

    Contains both input token stats (what tokens activate this component)
    and output token stats (what tokens this component predicts).
    """

    input: TokenPRLiftPMI  # Stats for input tokens
    output: TokenPRLiftPMI  # Stats for output (predicted) tokens


router = APIRouter(prefix="/api/correlations", tags=["correlations"])


# =============================================================================
# Interpretation Endpoint
# =============================================================================


class InterpretationHeadline(BaseModel):
    """Lightweight interpretation headline for bulk fetching."""

    label: str
    confidence: str
    detection_score: float | None = None
    fuzzing_score: float | None = None


class InterpretationDetail(BaseModel):
    """Full interpretation detail fetched on-demand."""

    reasoning: str
    prompt: str


@router.get("/interpretations")
@log_errors
def get_all_interpretations(
    loaded: DepLoadedRun,
) -> dict[str, InterpretationHeadline]:
    """Get all interpretation headlines (label + confidence + eval scores).

    Returns a dict keyed by component_key (layer:cIdx).
    Returns empty dict if no interpretations are available.
    Reasoning and prompt are excluded - fetch individually via
    GET /interpretations/{layer}/{component_idx} when needed.
    """
    interpretations = loaded.interp.get_all_interpretations()
    if interpretations is None:
        return {}

    detection_scores = loaded.interp.get_detection_scores()
    fuzzing_scores = loaded.interp.get_fuzzing_scores()

    return {
        _concrete_to_canonical_key(key, loaded.topology): InterpretationHeadline(
            label=result.label,
            confidence=result.confidence,
            detection_score=detection_scores.get(key) if detection_scores else None,
            fuzzing_score=fuzzing_scores.get(key) if fuzzing_scores else None,
        )
        for key, result in interpretations.items()
    }


@router.get("/interpretations/{layer}/{component_idx}")
@log_errors
def get_interpretation_detail(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
) -> InterpretationDetail:
    """Get the full interpretation detail (reasoning + prompt).

    Returns reasoning and prompt for the specified component.
    """
    concrete_key = _canonical_to_concrete_key(layer, component_idx, loaded.topology)
    result = loaded.interp.get_interpretation(concrete_key)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No interpretation found for component {layer}:{component_idx}",
        )

    return InterpretationDetail(reasoning=result.reasoning, prompt=result.prompt)


@router.post("/interpretations/{layer}/{component_idx}")
@log_errors
async def request_component_interpretation(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
) -> InterpretationHeadline:
    """Generate an interpretation for a component on-demand.

    Requires OPENROUTER_API_KEY environment variable.
    Returns the headline (label + confidence). Full detail available via GET endpoint.
    """
    import os

    from openrouter import OpenRouter

    from spd.autointerp.config import CompactSkepticalConfig
    from spd.autointerp.interpret import (
        get_architecture_info,
        interpret_component,
    )
    from spd.autointerp.llm_api import CostTracker, LLMClient, RateLimiter  # noqa: F811

    component_key = _canonical_to_concrete_key(layer, component_idx, loaded.topology)
    run_id = loaded.harvest.run_id

    existing = loaded.interp.get_interpretation(component_key)
    if existing is not None:
        return InterpretationHeadline(
            label=existing.label,
            confidence=existing.confidence,
        )

    component_data = loaded.harvest.get_component(component_key)
    assert component_data is not None, f"Component {component_key} not found in harvest"

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable not set",
        )

    arch = get_architecture_info(loaded.run.wandb_path)

    token_stats = loaded.harvest.get_token_stats()
    assert token_stats is not None, "Token stats required for interpretation"

    input_token_stats = analysis.get_input_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k=20
    )
    output_token_stats = analysis.get_output_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k=50
    )
    if input_token_stats is None or output_token_stats is None:
        raise HTTPException(
            status_code=400,
            detail=f"Token stats not available for component {component_key}",
        )

    config = CompactSkepticalConfig(
        model="google/gemini-3-flash-preview",
        reasoning_effort=None,
    )

    ci_threshold = loaded.harvest.get_ci_threshold()

    async with OpenRouter(api_key=api_key) as api:
        llm = LLMClient(
            api=api,
            rate_limiter=RateLimiter(max_requests=10),
            cost_tracker=CostTracker(),
        )
        try:
            result = await interpret_component(
                llm=llm,
                config=config,
                component=component_data,
                arch=arch,
                app_tok=loaded.tokenizer,
                input_token_stats=input_token_stats,
                output_token_stats=output_token_stats,
                ci_threshold=ci_threshold,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate interpretation: {e}",
            ) from e

    loaded.interp.save_interpretation(result)

    logger.info(f"Generated interpretation for {component_key}: {result.label}")

    return InterpretationHeadline(
        label=result.label,
        confidence=result.confidence,
    )


@router.get("/intruder_scores")
@log_errors
def get_intruder_scores(loaded: DepLoadedRun) -> dict[str, float]:
    """Get intruder eval scores for all components.

    Returns a dict keyed by component_key (layer:cIdx) → score (0-1).
    Returns empty dict if no intruder scores are available.
    """
    return loaded.interp.get_intruder_scores() or {}


# =============================================================================
# Component Correlation Data Endpoints
# =============================================================================


@router.get("/token_stats/{layer}/{component_idx}")
@log_errors
def get_component_token_stats(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1)],
) -> TokenStatsResponse | None:
    """Get token precision/recall/lift/PMI for a component.

    Returns stats for both input tokens (what activates this component)
    and output tokens (what this component predicts).
    Returns None if token stats haven't been harvested for this run.
    """
    token_stats = loaded.harvest.get_token_stats()
    component_key = _canonical_to_concrete_key(layer, component_idx, loaded.topology)

    input_stats = analysis.get_input_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k
    )
    output_stats = analysis.get_output_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k
    )

    if input_stats is None or output_stats is None:
        return None

    return TokenStatsResponse(
        input=TokenPRLiftPMI(
            top_recall=input_stats.top_recall,
            top_precision=input_stats.top_precision,
            top_lift=input_stats.top_lift,
            top_pmi=input_stats.top_pmi,
            bottom_pmi=input_stats.bottom_pmi,
        ),
        output=TokenPRLiftPMI(
            top_recall=output_stats.top_recall,
            top_precision=output_stats.top_precision,
            top_lift=output_stats.top_lift,
            top_pmi=output_stats.top_pmi,
            bottom_pmi=output_stats.bottom_pmi,
        ),
    )


class BulkCorrelationsRequest(BaseModel):
    """Request for bulk correlations."""

    component_keys: list[str]
    top_k: int = 20  # Sufficient for initial display


class BulkTokenStatsRequest(BaseModel):
    """Request for bulk token stats."""

    component_keys: list[str]
    top_k: int = 30  # Sufficient for initial display


@router.post("/components/bulk")
@log_errors
def get_component_correlations_bulk(
    request: BulkCorrelationsRequest,
    loaded: DepLoadedRun,
) -> dict[str, ComponentCorrelationsResponse]:
    """Bulk fetch correlations for multiple components.

    Returns a dict keyed by component_key. Components not found are omitted.
    """
    correlations = loaded.harvest.get_correlations()

    def to_schema(c: analysis.CorrelatedComponent) -> CorrelatedComponent:
        return CorrelatedComponent(
            component_key=c.component_key,
            score=c.score,
            count_i=c.count_i,
            count_j=c.count_j,
            count_ij=c.count_ij,
            n_tokens=c.count_total,
        )

    def to_concrete(canonical_key: str) -> str:
        layer, idx = canonical_key.rsplit(":", 1)
        return _canonical_to_concrete_key(layer, int(idx), loaded.topology)

    result: dict[str, ComponentCorrelationsResponse] = {}

    for canonical_key in request.component_keys:
        concrete_key = to_concrete(canonical_key)
        if not analysis.has_component(correlations, concrete_key):
            continue

        result[canonical_key] = ComponentCorrelationsResponse(
            precision=[
                to_schema(c)
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "precision", request.top_k
                )
            ],
            recall=[
                to_schema(c)
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "recall", request.top_k
                )
            ],
            jaccard=[
                to_schema(c)
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "jaccard", request.top_k
                )
            ],
            pmi=[
                to_schema(c)
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "pmi", request.top_k
                )
            ],
            bottom_pmi=[
                to_schema(c)
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "pmi", request.top_k, largest=False
                )
            ],
        )

    return result


@router.post("/token_stats/bulk")
@log_errors
def get_component_token_stats_bulk(
    request: BulkTokenStatsRequest,
    loaded: DepLoadedRun,
) -> dict[str, TokenStatsResponse]:
    """Bulk fetch token stats for multiple components.

    Returns a dict keyed by component_key. Components not found are omitted.
    """
    token_stats = loaded.harvest.get_token_stats()
    result: dict[str, TokenStatsResponse] = {}

    def to_concrete(canonical_key: str) -> str:
        layer, idx = canonical_key.rsplit(":", 1)
        return _canonical_to_concrete_key(layer, int(idx), loaded.topology)

    for canonical_key in request.component_keys:
        concrete_key = to_concrete(canonical_key)
        input_stats = analysis.get_input_token_stats(
            token_stats, concrete_key, loaded.tokenizer, request.top_k
        )
        output_stats = analysis.get_output_token_stats(
            token_stats, concrete_key, loaded.tokenizer, request.top_k
        )

        if input_stats is None or output_stats is None:
            continue

        result[canonical_key] = TokenStatsResponse(
            input=TokenPRLiftPMI(
                top_recall=input_stats.top_recall,
                top_precision=input_stats.top_precision,
                top_lift=input_stats.top_lift,
                top_pmi=input_stats.top_pmi,
                bottom_pmi=input_stats.bottom_pmi,
            ),
            output=TokenPRLiftPMI(
                top_recall=output_stats.top_recall,
                top_precision=output_stats.top_precision,
                top_lift=output_stats.top_lift,
                top_pmi=output_stats.top_pmi,
                bottom_pmi=output_stats.bottom_pmi,
            ),
        )

    return result


@router.get("/components/{layer}/{component_idx}")
@log_errors
def get_component_correlations(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1)],
) -> ComponentCorrelationsResponse:
    """Get correlated components for a specific component.

    Returns top-k correlations across different metrics (precision, recall, Jaccard, PMI).
    Returns None if correlations haven't been harvested for this run.
    """
    correlations = loaded.harvest.get_correlations()
    component_key = _canonical_to_concrete_key(layer, component_idx, loaded.topology)

    if not analysis.has_component(correlations, component_key):
        raise HTTPException(
            status_code=404, detail=f"Component {component_key} not found in correlations"
        )

    def to_schema(c: analysis.CorrelatedComponent) -> CorrelatedComponent:
        return CorrelatedComponent(
            component_key=c.component_key,
            score=c.score,
            count_i=c.count_i,
            count_j=c.count_j,
            count_ij=c.count_ij,
            n_tokens=c.count_total,
        )

    return ComponentCorrelationsResponse(
        precision=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "precision", top_k
            )
        ],
        recall=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "recall", top_k
            )
        ],
        jaccard=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "jaccard", top_k
            )
        ],
        pmi=[
            to_schema(c)
            for c in analysis.get_correlated_components(correlations, component_key, "pmi", top_k)
        ],
        bottom_pmi=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "pmi", top_k, largest=False
            )
        ],
    )
