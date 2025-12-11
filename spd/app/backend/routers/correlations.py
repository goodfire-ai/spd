"""Component correlation endpoints.

These endpoints serve data produced by the SLURM batch job (harvest_correlations.py),
which computes component co-occurrence statistics and token associations.
Also includes job management endpoints for submitting and monitoring harvest jobs.
"""

import time
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.lib.component_correlations import (
    ComponentCorrelations,
    ComponentTokenStats,
    get_correlations_path,
    get_token_stats_path,
)
from spd.app.backend.lib.component_correlations import (
    CorrelatedComponentWithCounts as CorrelatedComponentDC,
)
from spd.app.backend.lib.component_correlations_slurm import (
    CompletedStatus,
    FailedStatus,
    PendingStatus,
    RunningStatus,
    get_last_log_line,
    read_job_state,
    status_file_exists,
    submit_correlation_job,
)
from spd.app.backend.utils import log_errors
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
    f1: list[CorrelatedComponent]
    jaccard: list[CorrelatedComponent]
    pmi: list[CorrelatedComponent]


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
# Job Management Schemas & Endpoints (must be before wildcard routes)
# =============================================================================


class HarvestParamsResponse(BaseModel):
    n_batches: int
    batch_size: int
    context_length: int
    ci_threshold: float


class PendingStatusResponse(BaseModel):
    status: Literal["pending"]
    job_id: str
    submitted_at: str
    params: HarvestParamsResponse
    last_log_line: str | None


class RunningStatusResponse(BaseModel):
    status: Literal["running"]
    job_id: str
    submitted_at: str
    params: HarvestParamsResponse
    last_log_line: str | None


class CompletedStatusResponse(BaseModel):
    status: Literal["completed"]
    job_id: str
    submitted_at: str
    params: HarvestParamsResponse
    n_tokens: int
    n_components: int


class FailedStatusResponse(BaseModel):
    status: Literal["failed"]
    job_id: str
    submitted_at: str
    params: HarvestParamsResponse
    error: str


CorrelationJobStatusResponse = Annotated[
    PendingStatusResponse | RunningStatusResponse | CompletedStatusResponse | FailedStatusResponse,
    "Discriminated union on 'status' field",
]


class SubmitJobResponse(BaseModel):
    job_id: str
    status: Literal["pending"]


@router.get("/jobs/status")
@log_errors
def get_job_status(loaded: DepLoadedRun) -> CorrelationJobStatusResponse:
    """Get the correlation job status for the currently loaded run.

    Returns 404 if no job has been submitted yet.
    """
    _, _, run_id = parse_wandb_run_path(loaded.run.wandb_path)

    if not status_file_exists(run_id):
        raise HTTPException(status_code=404, detail="No correlation job for this run")

    state = read_job_state(run_id)
    params = HarvestParamsResponse(
        n_batches=state.params.n_batches,
        batch_size=state.params.batch_size,
        context_length=state.params.context_length,
        ci_threshold=state.params.ci_threshold,
    )

    match state.job_status:
        case CompletedStatus(n_tokens=n_tokens, n_components=n_components):
            return CompletedStatusResponse(
                status="completed",
                job_id=state.job_id,
                submitted_at=state.submitted_at,
                params=params,
                n_tokens=n_tokens,
                n_components=n_components,
            )
        case FailedStatus(error=error):
            return FailedStatusResponse(
                status="failed",
                job_id=state.job_id,
                submitted_at=state.submitted_at,
                params=params,
                error=error,
            )
        case PendingStatus():
            return PendingStatusResponse(
                status="pending",
                job_id=state.job_id,
                submitted_at=state.submitted_at,
                params=params,
                last_log_line=get_last_log_line(run_id, state.job_id),
            )
        case RunningStatus():
            return RunningStatusResponse(
                status="running",
                job_id=state.job_id,
                submitted_at=state.submitted_at,
                params=params,
                last_log_line=get_last_log_line(run_id, state.job_id),
            )


@router.post("/jobs/submit")
@log_errors
def submit_job(loaded: DepLoadedRun) -> SubmitJobResponse:
    """Submit a SLURM job to harvest correlations for the currently loaded run."""
    wandb_path = loaded.run.wandb_path
    _, _, run_id = parse_wandb_run_path(wandb_path)

    if status_file_exists(run_id):
        state = read_job_state(run_id)
        if state.job_status.status in ("pending", "running"):
            raise HTTPException(
                status_code=400,
                detail=f"Job already {state.job_status.status} (job_id: {state.job_id})",
            )
        if state.job_status.status == "completed":
            raise HTTPException(status_code=400, detail="Correlations already computed")

    job_id = submit_correlation_job(wandb_path, run_id)
    return SubmitJobResponse(job_id=job_id, status="pending")


# =============================================================================
# Component Correlation Data Endpoints
# =============================================================================


def _get_correlations(run_id: str) -> ComponentCorrelations | None:
    """Load correlations from disk."""
    start = time.perf_counter()

    path = get_correlations_path(run_id)
    if not path.exists():
        logger.warning(f"Correlations not found at {path}")
        return None

    correlations = ComponentCorrelations.load(path)
    load_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Loaded correlations for {run_id} in {load_ms:.1f}ms")
    return correlations


def _get_token_stats(run_id: str) -> ComponentTokenStats | None:
    """Load token stats from disk."""
    start = time.perf_counter()

    path = get_token_stats_path(run_id)
    if not path.exists():
        return None

    token_stats = ComponentTokenStats.load(path)
    load_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Loaded token stats for {run_id} in {load_ms:.1f}ms")
    return token_stats


@router.get("/{layer}/{component_idx}")
@log_errors
def get_component_correlations(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> ComponentCorrelationsResponse | None:
    """Get correlated components for a specific component.

    Returns top-k correlations across different metrics (precision, recall, F1, Jaccard, PMI).
    Returns None if correlations haven't been harvested for this run.
    """
    start = time.perf_counter()

    run_id = loaded.run.wandb_path.split("/")[-1]
    correlations = _get_correlations(run_id)

    if correlations is None:
        return None

    component_key = f"{layer}:{component_idx}"

    if component_key not in correlations.component_keys:
        raise HTTPException(
            status_code=404, detail=f"Component {component_key} not found in correlations"
        )

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
        f1=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "f1", top_k)
        ],
        jaccard=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "jaccard", top_k)
        ],
        pmi=[
            to_schema(c)
            for c in correlations.get_correlated_with_counts(component_key, "pmi", top_k)
        ],
    )

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(f"get_component_correlations: {component_key} in {total_ms:.1f}ms")
    return response


@router.get("/token_stats/{layer}/{component_idx}")
@log_errors
def get_component_token_stats(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> TokenStatsResponse | None:
    """Get token precision/recall/lift/PMI for a component.

    Returns stats for both input tokens (what activates this component)
    and output tokens (what this component predicts).
    Returns None if token stats haven't been harvested for this run.
    """
    start = time.perf_counter()

    run_id = loaded.run.wandb_path.split("/")[-1]
    token_stats = _get_token_stats(run_id)

    if token_stats is None:
        return None

    component_key = f"{layer}:{component_idx}"

    input_stats = token_stats.get_input_tok_stats(component_key, loaded.tokenizer, top_k=top_k)
    output_stats = token_stats.get_output_tok_stats(component_key, loaded.tokenizer, top_k=top_k)

    if input_stats is None or output_stats is None:
        return None

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(f"get_component_token_stats: {component_key} in {total_ms:.1f}ms")

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


def _format_token_list(tokens: list[tuple[str, float]], label: str, max_items: int = 10) -> str:
    """Format a list of (token, score) tuples as a readable string."""
    if not tokens:
        return f"{label}: (none)"
    items = [f"{repr(tok)}={score:.3f}" for tok, score in tokens[:max_items]]
    return f"{label}: {', '.join(items)}"


def _format_example(tokens: list[str], ci_values: list[float], active_pos: int) -> str:
    """Format a single activation example with the active token highlighted."""
    parts = []
    for i, (tok, ci) in enumerate(zip(tokens, ci_values, strict=True)):
        if i == active_pos:
            parts.append(f">>>{tok}<<<[{ci:.2f}]")
        elif ci > 0.1:
            parts.append(f"{tok}[{ci:.2f}]")
        else:
            parts.append(tok)
    return "".join(parts)


@router.get("/interpret/{layer}/{component_idx}", response_class=PlainTextResponse)
@log_errors
def get_component_interpretation(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    manager: DepStateManager,
    top_k: Annotated[int, Query(ge=1, le=20)] = 10,
) -> str:
    """Get a text-formatted interpretation report for a component.

    Combines activation examples, token statistics, and correlations into
    a single readable report suitable for LLM analysis.

    Returns plain text (not JSON) for easy reading in terminals or by LLMs.
    """
    lines: list[str] = []
    component_key = f"{layer}:{component_idx}"
    run_id = loaded.run.wandb_path.split("/")[-1]

    lines.append(f"# Component Interpretation Report: {component_key}")
    lines.append(f"Run: {loaded.run.wandb_path}")
    lines.append("")

    # --- Section 1: Activation Examples ---
    lines.append("## Activation Examples")
    lines.append("Contexts where this component fires strongly (>>>token<<< = active position):")
    lines.append("")

    detail = manager.db.get_component_activation_context_detail(
        loaded.run.id, loaded.context_length, layer, component_idx
    )
    if detail is None:
        lines.append("(No activation contexts available - run harvest first)")
    else:
        lines.append(f"Mean CI: {detail.mean_ci:.4f}")
        lines.append("")
        for i, (tokens, ci_vals, active_pos) in enumerate(
            zip(
                detail.example_tokens,
                detail.example_ci,
                detail.example_active_pos,
                strict=True,
            )
        ):
            example_str = _format_example(tokens, ci_vals, active_pos)
            lines.append(f"  {i + 1}. {example_str}")
        lines.append("")

    # --- Section 2: Token Statistics ---
    lines.append("## Token Statistics")
    lines.append("What tokens activate this component (input) and what it predicts (output).")
    lines.append("")

    token_stats = _get_token_stats(run_id)
    if token_stats is None:
        lines.append("(No token stats available - run correlation harvest job first)")
    else:
        input_stats = token_stats.get_input_tok_stats(component_key, loaded.tokenizer, top_k=top_k)
        output_stats = token_stats.get_output_tok_stats(
            component_key, loaded.tokenizer, top_k=top_k
        )

        if input_stats:
            lines.append("### Input Tokens (what activates this component)")
            lines.append(_format_token_list(input_stats.top_pmi, "Top PMI (most specific)"))
            lines.append(_format_token_list(input_stats.top_precision, "Top Precision"))
            lines.append(_format_token_list(input_stats.top_recall, "Top Recall (most frequent)"))
            lines.append("")

        if output_stats:
            lines.append("### Output Tokens (what this component predicts)")
            lines.append(_format_token_list(output_stats.top_pmi, "Top PMI (most boosted)"))
            lines.append(_format_token_list(output_stats.top_precision, "Top Precision"))
            if output_stats.bottom_pmi:
                lines.append(
                    _format_token_list(output_stats.bottom_pmi, "Bottom PMI (most suppressed)")
                )
            lines.append("")

    # --- Section 3: Correlated Components ---
    lines.append("## Correlated Components")
    lines.append("Other components that fire together with this one.")
    lines.append("")

    correlations = _get_correlations(run_id)
    if correlations is None:
        lines.append("(No correlations available - run correlation harvest job first)")
    elif component_key not in correlations.component_keys:
        lines.append(f"(Component {component_key} not found in correlations)")
    else:
        pmi_corr = correlations.get_correlated(component_key, "pmi", top_k)
        f1_corr = correlations.get_correlated(component_key, "f1", top_k)

        if pmi_corr:
            pmi_str = ", ".join([f"{c.component_key}={c.score:.2f}" for c in pmi_corr[:5]])
            lines.append(f"Top PMI: {pmi_str}")
        if f1_corr:
            f1_str = ", ".join([f"{c.component_key}={c.score:.2f}" for c in f1_corr[:5]])
            lines.append(f"Top F1: {f1_str}")
        lines.append("")

    return "\n".join(lines)
