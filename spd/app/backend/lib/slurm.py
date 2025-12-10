"""Simple SLURM job submission for the app.

This is a stripped-down version of compute_utils.py, designed for single-task
jobs (like correlation harvesting) without the complexity of job arrays,
git snapshots, or multi-node DDP.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from spd.app.backend.db.database import CORRELATIONS_DIR
from spd.settings import DEFAULT_PARTITION_NAME, REPO_ROOT


@dataclass
class HarvestParams:
    """Parameters for correlation harvesting."""

    n_batches: int = 100
    batch_size: int = 256
    context_length: int = 512
    ci_threshold: float = 1e-6


DEFAULT_HARVEST_PARAMS = HarvestParams()


# Discriminated union for job status
@dataclass
class PendingStatus:
    status: Literal["pending"] = "pending"


@dataclass
class RunningStatus:
    status: Literal["running"] = "running"


@dataclass
class CompletedStatus:
    status: Literal["completed"] = "completed"
    n_tokens: int = 0
    n_components: int = 0


@dataclass
class FailedStatus:
    status: Literal["failed"] = "failed"
    error: str = ""


JobStatus = PendingStatus | RunningStatus | CompletedStatus | FailedStatus


@dataclass
class CorrelationJobState:
    """Full state of a correlation harvesting job."""

    job_id: str
    submitted_at: str
    params: HarvestParams
    job_status: JobStatus


def get_status_path(run_id: str) -> Path:
    """Get the path to the status file for a run."""

    return CORRELATIONS_DIR / run_id / "status.json"


def get_log_path(run_id: str, job_id: str) -> Path:
    """Get the path to the SLURM log file."""
    return Path.home() / "slurm_logs" / f"correlations-{run_id}-{job_id}.out"


def get_last_log_line(run_id: str, job_id: str) -> str | None:
    """Get the last non-empty line from the SLURM log file."""
    log_path = get_log_path(run_id, job_id)
    if not log_path.exists():
        return None
    text = log_path.read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def status_file_exists(run_id: str) -> bool:
    """Check if a status file exists for the run."""
    return get_status_path(run_id).exists()


def read_job_state(run_id: str) -> CorrelationJobState:
    """Read the job state from the status file.

    Caller should check status_file_exists() first.
    """
    status_path = get_status_path(run_id)
    data = json.loads(status_path.read_text())
    params_data = data.get("params", {})

    status_str = data["status"]
    job_status: JobStatus
    match status_str:
        case "pending":
            job_status = PendingStatus()
        case "running":
            job_status = RunningStatus()
        case "completed":
            job_status = CompletedStatus(
                n_tokens=data.get("n_tokens", 0),
                n_components=data.get("n_components", 0),
            )
        case "failed":
            job_status = FailedStatus(error=data.get("error", ""))
        case _:
            job_status = FailedStatus(error=f"Unknown status: {status_str}")

    return CorrelationJobState(
        job_id=data["job_id"],
        submitted_at=data["submitted_at"],
        params=HarvestParams(
            n_batches=params_data.get("n_batches", 500),
            batch_size=params_data.get("batch_size", 32),
            context_length=params_data.get("context_length", 128),
            ci_threshold=params_data.get("ci_threshold", 1e-6),
        ),
        job_status=job_status,
    )


def write_job_state(run_id: str, state: CorrelationJobState) -> None:
    """Write job state to the status file."""
    status_path = get_status_path(run_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, str | int | dict[str, int | float]] = {
        "status": state.job_status.status,
        "job_id": state.job_id,
        "submitted_at": state.submitted_at,
        "params": {
            "n_batches": state.params.n_batches,
            "batch_size": state.params.batch_size,
            "context_length": state.params.context_length,
            "ci_threshold": state.params.ci_threshold,
        },
    }

    match state.job_status:
        case CompletedStatus(n_tokens=n_tokens, n_components=n_components):
            data["n_tokens"] = n_tokens
            data["n_components"] = n_components
        case FailedStatus(error=error):
            data["error"] = error
        case PendingStatus() | RunningStatus():
            pass

    status_path.write_text(json.dumps(data, indent=2))


def submit_correlation_job(
    wandb_path: str,
    run_id: str,
    partition: str = DEFAULT_PARTITION_NAME,
) -> str:
    """Submit a SLURM job to harvest correlations.

    Args:
        wandb_path: Full W&B path (entity/project/run_id)
        run_id: Just the run ID portion (for file naming)
        partition: SLURM partition to use

    Returns:
        The SLURM job ID
    """
    status_path = get_status_path(run_id)
    logs_dir = Path.home() / "slurm_logs"
    logs_dir.mkdir(exist_ok=True)
    scripts_dir = Path.home() / "sbatch_scripts"
    scripts_dir.mkdir(exist_ok=True)

    params = DEFAULT_HARVEST_PARAMS

    script_content = f"""\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition={partition}
#SBATCH --time=4:00:00
#SBATCH --job-name=spd-corr-{run_id[:8]}
#SBATCH --output={logs_dir}/correlations-{run_id}-%j.out

source {REPO_ROOT}/.venv/bin/activate
cd {REPO_ROOT}

python -m spd.app.scripts.harvest_correlations \\
    "{wandb_path}" \\
    --n_batches {params.n_batches} \\
    --batch_size {params.batch_size} \\
    --context_length {params.context_length} \\
    --ci_threshold {params.ci_threshold} \\
    --status_file "{status_path}"
"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = scripts_dir / f"correlations_{run_id}_{timestamp}.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit SLURM job: {result.stderr}")

    # Extract job ID from sbatch output (format: "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    now = datetime.now().isoformat()

    write_job_state(
        run_id,
        CorrelationJobState(
            job_id=job_id,
            submitted_at=now,
            params=params,
            job_status=PendingStatus(),
        ),
    )

    # Rename script to include job ID for easier correlation
    final_script_path = scripts_dir / f"correlations_{job_id}.sh"
    script_path.rename(final_script_path)

    return job_id
