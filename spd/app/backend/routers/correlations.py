"""Component correlation and interpretation endpoints.

These endpoints serve data produced by the harvest pipeline (spd.autointerp.harvest),
which computes component co-occurrence statistics, token associations, and interpretations.
"""

import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.lib.component_correlations import (
    ComponentCorrelations,
    ComponentTokenStats,
    get_correlations_path,
    get_token_stats_path,
)
from spd.app.backend.lib.component_correlations import (
    CorrelatedComponentWithCounts as CorrelatedComponentDC,
)
from spd.app.backend.utils import log_errors, timer
from spd.autointerp.loaders import load_interpretations
from spd.log import logger
from spd.utils.wandb_utils import parse_wandb_run_path


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


class InterpretationResponse(BaseModel):
    """Interpretation label for a component."""

    label: str
    confidence: str
    reasoning: str


@router.get("/interpretations/{layer}/{component_idx}")
@log_errors
def get_component_interpretation(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
) -> InterpretationResponse | None:
    """Get interpretation label for a component.

    Returns None if no interpretation exists for this component.
    """
    _, _, run_id = parse_wandb_run_path(loaded.run.wandb_path)

    interpretations = load_interpretations(run_id)
    if interpretations is None:
        return None

    component_key = f"{layer}:{component_idx}"
    result = interpretations.get(component_key)
    if result is None:
        return None

    return InterpretationResponse(
        label=result.label,
        confidence=result.confidence,
        reasoning=result.reasoning,
    )


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
    run_id = loaded.run.wandb_path.split("/")[-1]

    path = get_token_stats_path(run_id)
    if not path.exists():
        return None

    with timer(f"Loading token stats for {run_id}"):
        token_stats = ComponentTokenStats.load(path)

    component_key = f"{layer}:{component_idx}"

    with timer(f"Getting input token stats for {component_key}"):
        input_stats = token_stats.get_input_tok_stats(component_key, loaded.tokenizer, top_k=top_k)

    with timer(f"Getting output token stats for {component_key}"):
        output_stats = token_stats.get_output_tok_stats(
            component_key, loaded.tokenizer, top_k=top_k
        )

    if input_stats is None or output_stats is None:
        return None

    assert input_stats.bottom_pmi is None, "Input stats should not have bottom PMI"
    assert output_stats.bottom_pmi is not None, "Output stats should have bottom PMI"

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


@router.get("/components/{layer}/{component_idx}")
@log_errors
def get_component_correlations(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1)],
) -> ComponentCorrelationsResponse | None:
    """Get correlated components for a specific component.

    Returns top-k correlations across different metrics (precision, recall, Jaccard, PMI).
    Returns None if correlations haven't been harvested for this run.
    """
    start = time.perf_counter()

    run_id = loaded.run.wandb_path.split("/")[-1]

    path = get_correlations_path(run_id)
    if not path.exists():
        return None

    with timer(f"Loading correlations for {run_id}"):
        correlations = ComponentCorrelations.load(path)

    component_key = f"{layer}:{component_idx}"

    if component_key not in correlations.component_keys:
        raise HTTPException(
            status_code=404, detail=f"Component {component_key} not found in correlations"
        )

    # annoying thing we have to do because we have separate model objects and pydantic DTOs
    def to_schema(c: CorrelatedComponentDC) -> CorrelatedComponent:
        return CorrelatedComponent(
            component_key=c.component_key,
            score=c.score,
            count_i=c.count_i,
            count_j=c.count_j,
            count_ij=c.count_ij,
            n_tokens=c.count_total,
        )

    response = ComponentCorrelationsResponse(
        precision=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "precision", top_k)
        ],
        recall=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "recall", top_k)
        ],
        jaccard=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "jaccard", top_k)
        ],
        pmi=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "pmi", top_k)
        ],
        bottom_pmi=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(
                component_key, "pmi", top_k, largest=False
            )
        ],
    )

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(f"get_component_correlations: {component_key} in {total_ms:.1f}ms")
    return response
