"""Job manifest with atomic state tracking for fault-tolerant batch processing.

The manifest tracks the status of each prompt in a batch job, enabling:
- Checkpoint/resume: Skip completed prompts on restart
- Fault tolerance: Atomic updates prevent corruption
- Progress monitoring: Real-time status across all tasks
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock


@dataclass
class PromptStatus:
    """Status of a single prompt in the batch."""

    idx: int
    prompt: str
    answer_prefix: str = ""
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    task_id: int | None = None
    output_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptStatus":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JobManifest:
    """Atomic job manifest for tracking batch processing state.

    All state updates are atomic using file locking to prevent corruption
    when multiple SLURM tasks update the manifest concurrently.

    Usage:
        # Create a new job
        manifest = JobManifest(job_dir)
        manifest.create(prompts, config, n_tasks=100)

        # In worker: get pending prompts for this task
        pending = manifest.get_pending_for_task(task_id)
        for p in pending:
            manifest.update_status(p.idx, "running")
            # ... process ...
            manifest.update_status(p.idx, "completed", output_path="...")

        # Monitor progress
        progress = manifest.get_progress()
        print(f"Completed: {progress['completed']}/{progress['total']}")
    """

    job_dir: Path
    _lock_timeout: float = 30.0

    def __post_init__(self):
        self.job_dir = Path(self.job_dir)
        self._manifest_path = self.job_dir / "manifest.json"
        self._lock_path = self.job_dir / "manifest.lock"

    def _read(self) -> dict[str, Any]:
        """Read manifest without locking (internal use)."""
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self._manifest_path}")
        with open(self._manifest_path) as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        """Write manifest without locking (internal use)."""
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def _atomic_update(self, update_fn) -> Any:
        """Perform atomic read-modify-write with file locking."""
        with FileLock(self._lock_path, timeout=self._lock_timeout):
            data = self._read()
            result = update_fn(data)
            self._write(data)
            return result

    def create(
        self,
        prompts: list[dict[str, str]],
        config: dict[str, Any],
        n_tasks: int,
        job_id: str | None = None,
    ) -> str:
        """Create a new job manifest.

        Args:
            prompts: List of dicts with 'prompt' and optional 'answer_prefix'
            config: Pipeline configuration dict
            n_tasks: Number of SLURM array tasks
            job_id: Optional job ID (auto-generated if not provided)

        Returns:
            The job ID
        """
        self.job_dir.mkdir(parents=True, exist_ok=True)

        if job_id is None:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Assign prompts to tasks round-robin
        prompt_statuses = []
        for idx, p in enumerate(prompts):
            task_id = idx % n_tasks
            prompt_statuses.append(
                PromptStatus(
                    idx=idx,
                    prompt=p["prompt"],
                    answer_prefix=p.get("answer_prefix", ""),
                    task_id=task_id,
                ).to_dict()
            )

        manifest_data = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "n_tasks": n_tasks,
            "status": "created",
            "config": config,
            "prompts": prompt_statuses,
            "llm_rate_limit": config.get("llm_rate_limit", 3000),  # Global rate limit across all workers
        }

        self._write(manifest_data)
        return job_id

    def update_status(
        self,
        idx: int,
        status: Literal["pending", "running", "completed", "failed"],
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Atomically update the status of a prompt.

        Args:
            idx: Prompt index
            status: New status
            output_path: Path to output file (for completed)
            error: Error message (for failed)
        """

        def update(data):
            prompt = data["prompts"][idx]
            prompt["status"] = status

            now = datetime.now().isoformat()
            if status == "running":
                prompt["started_at"] = now
            elif status in ("completed", "failed"):
                prompt["completed_at"] = now
                if prompt.get("started_at"):
                    start = datetime.fromisoformat(prompt["started_at"])
                    end = datetime.fromisoformat(now)
                    prompt["duration_seconds"] = (end - start).total_seconds()

            if output_path is not None:
                prompt["output_path"] = output_path
            if error is not None:
                prompt["error"] = error

        self._atomic_update(update)

    def get_pending_for_task(self, task_id: int) -> list[PromptStatus]:
        """Get all pending or failed prompts assigned to a specific task.

        This enables resume functionality: when a task restarts, it only
        processes prompts that haven't completed yet.

        Args:
            task_id: SLURM array task ID

        Returns:
            List of PromptStatus for prompts needing processing
        """
        data = self._read()
        pending = []
        for p in data["prompts"]:
            if p["task_id"] == task_id and p["status"] in ("pending", "failed"):
                pending.append(PromptStatus.from_dict(p))
        return pending

    def get_all_for_task(self, task_id: int) -> list[PromptStatus]:
        """Get all prompts assigned to a specific task (any status)."""
        data = self._read()
        return [
            PromptStatus.from_dict(p)
            for p in data["prompts"]
            if p["task_id"] == task_id
        ]

    def get_progress(self) -> dict[str, Any]:
        """Get current progress statistics.

        Returns:
            Dict with keys: total, pending, running, completed, failed,
            percent_complete, avg_duration_seconds
        """
        data = self._read()

        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        durations = []

        for p in data["prompts"]:
            counts[p["status"]] += 1
            if p.get("duration_seconds"):
                durations.append(p["duration_seconds"])

        total = data["total_prompts"]
        completed = counts["completed"]

        return {
            "job_id": data["job_id"],
            "total": total,
            "pending": counts["pending"],
            "running": counts["running"],
            "completed": completed,
            "failed": counts["failed"],
            "percent_complete": (completed / total * 100) if total > 0 else 0,
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
        }

    def get_failed_prompts(self) -> list[PromptStatus]:
        """Get all failed prompts with their error messages."""
        data = self._read()
        return [
            PromptStatus.from_dict(p) for p in data["prompts"] if p["status"] == "failed"
        ]

    def get_completed_outputs(self) -> list[str]:
        """Get paths to all completed output files."""
        data = self._read()
        return [
            p["output_path"]
            for p in data["prompts"]
            if p["status"] == "completed" and p.get("output_path")
        ]

    def mark_job_status(self, status: str) -> None:
        """Update the overall job status."""

        def update(data):
            data["status"] = status
            if status == "running":
                data["started_at"] = datetime.now().isoformat()
            elif status in ("completed", "failed"):
                data["finished_at"] = datetime.now().isoformat()

        self._atomic_update(update)

    def get_config(self) -> dict[str, Any]:
        """Get the pipeline configuration stored in the manifest."""
        data = self._read()
        return data.get("config", {})

    def get_worker_rate_limit(self) -> int:
        """Get the per-worker LLM rate limit (total / n_tasks).

        This divides the global rate limit evenly across all workers
        to prevent hitting API rate limits when running distributed.
        """
        data = self._read()
        total_rate = data.get("llm_rate_limit", 3000)
        n_tasks = data.get("n_tasks", 1)
        # Each worker gets an equal share of the rate limit
        return max(1, total_rate // n_tasks)

    def exists(self) -> bool:
        """Check if the manifest file exists."""
        return self._manifest_path.exists()
