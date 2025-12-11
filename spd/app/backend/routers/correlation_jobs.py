"""Correlation harvesting job management endpoints."""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
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
from spd.utils.wandb_utils import parse_wandb_run_path

router = APIRouter(prefix="/api/correlation_jobs", tags=["correlation_jobs"])


class HarvestParamsResponse(BaseModel):
    n_batches: int
    batch_size: int
    context_length: int
    ci_threshold: float


# Discriminated union responses - all have job_id, submitted_at, params
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


@router.get("/status")
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


@router.post("/submit")
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
