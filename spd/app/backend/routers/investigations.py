"""Investigations endpoint for viewing agent swarm results.

Lists and serves investigation data from SPD_OUT_DIR/agent_swarm/.
Each task is treated as an independent investigation (flattened across swarms).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.settings import SPD_OUT_DIR

router = APIRouter(prefix="/api/investigations", tags=["investigations"])

SWARM_DIR = SPD_OUT_DIR / "agent_swarm"


class InvestigationSummary(BaseModel):
    """Summary of a single investigation (task)."""

    id: str  # swarm_id/task_id
    swarm_id: str
    task_id: int
    wandb_path: str | None
    created_at: str
    has_research_log: bool
    has_explanations: bool
    event_count: int
    last_event_time: str | None
    last_event_message: str | None
    # Agent-provided summary
    title: str | None
    summary: str | None
    status: str | None  # in_progress, completed, inconclusive


class EventEntry(BaseModel):
    """A single event from events.jsonl."""

    event_type: str
    timestamp: str
    message: str
    details: dict[str, Any] | None = None


class InvestigationDetail(BaseModel):
    """Full detail of an investigation including logs."""

    id: str
    swarm_id: str
    task_id: int
    wandb_path: str | None
    created_at: str
    research_log: str | None
    events: list[EventEntry]
    explanations: list[dict[str, Any]]
    artifact_ids: list[str]  # List of artifact IDs available for this investigation
    # Agent-provided summary
    title: str | None
    summary: str | None
    status: str | None


def _parse_swarm_metadata(swarm_path: Path) -> dict[str, Any] | None:
    """Parse metadata.json from a swarm directory."""
    metadata_path = swarm_path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(metadata_path.read_text())
        return data
    except Exception:
        return None


def _get_last_event(events_path: Path) -> tuple[str | None, str | None, int]:
    """Get the last event timestamp, message, and total count from events.jsonl."""
    if not events_path.exists():
        return None, None, 0

    last_time = None
    last_msg = None
    count = 0

    try:
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    event = json.loads(line)
                    last_time = event.get("timestamp")
                    last_msg = event.get("message")
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return last_time, last_msg, count


def _parse_task_summary(task_path: Path) -> tuple[str | None, str | None, str | None]:
    """Parse summary.json from a task directory. Returns (title, summary, status)."""
    summary_path = task_path / "summary.json"
    if not summary_path.exists():
        return None, None, None
    try:
        data: dict[str, Any] = json.loads(summary_path.read_text())
        return data.get("title"), data.get("summary"), data.get("status")
    except Exception:
        return None, None, None


def _list_artifact_ids(task_path: Path) -> list[str]:
    """List all artifact IDs for a task."""
    artifacts_dir = task_path / "artifacts"
    if not artifacts_dir.exists():
        return []
    artifact_ids = []
    for f in sorted(artifacts_dir.glob("graph_*.json")):
        artifact_ids.append(f.stem)  # e.g., "graph_001"
    return artifact_ids


def _get_task_created_at(task_path: Path, swarm_metadata: dict[str, Any] | None) -> str:
    """Get creation time for a task."""
    # Try to get from first event
    events_path = task_path / "events.jsonl"
    if events_path.exists():
        try:
            with open(events_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    event = json.loads(first_line)
                    if "timestamp" in event:
                        return event["timestamp"]
        except Exception:
            pass

    # Fall back to swarm metadata
    if swarm_metadata and "created_at" in swarm_metadata:
        return swarm_metadata["created_at"]

    # Fall back to directory mtime
    return datetime.fromtimestamp(task_path.stat().st_mtime).isoformat()


@router.get("")
def list_investigations() -> list[InvestigationSummary]:
    """List all investigations (tasks) flattened across swarms."""
    if not SWARM_DIR.exists():
        return []

    results = []

    for swarm_path in SWARM_DIR.iterdir():
        if not swarm_path.is_dir() or not swarm_path.name.startswith("swarm-"):
            continue

        swarm_id = swarm_path.name
        metadata = _parse_swarm_metadata(swarm_path)
        wandb_path = metadata.get("wandb_path") if metadata else None

        for task_path in swarm_path.iterdir():
            if not task_path.is_dir() or not task_path.name.startswith("task_"):
                continue

            try:
                task_id = int(task_path.name.split("_")[1])
            except (ValueError, IndexError):
                continue

            events_path = task_path / "events.jsonl"
            last_time, last_msg, event_count = _get_last_event(events_path)
            title, summary, status = _parse_task_summary(task_path)

            results.append(
                InvestigationSummary(
                    id=f"{swarm_id}/{task_id}",
                    swarm_id=swarm_id,
                    task_id=task_id,
                    wandb_path=wandb_path,
                    created_at=_get_task_created_at(task_path, metadata),
                    has_research_log=(task_path / "research_log.md").exists(),
                    has_explanations=(task_path / "explanations.jsonl").exists()
                    and (task_path / "explanations.jsonl").stat().st_size > 0,
                    event_count=event_count,
                    last_event_time=last_time,
                    last_event_message=last_msg,
                    title=title,
                    summary=summary,
                    status=status,
                )
            )

    # Sort by creation time, newest first
    results.sort(key=lambda x: x.created_at, reverse=True)
    return results


@router.get("/{swarm_id}/{task_id}")
def get_investigation(swarm_id: str, task_id: int) -> InvestigationDetail:
    """Get full details of an investigation."""
    swarm_path = SWARM_DIR / swarm_id
    task_path = swarm_path / f"task_{task_id}"

    if not task_path.exists() or not task_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Investigation {swarm_id}/{task_id} not found")

    metadata = _parse_swarm_metadata(swarm_path)

    # Read research log
    research_log = None
    research_log_path = task_path / "research_log.md"
    if research_log_path.exists():
        research_log = research_log_path.read_text()

    # Read events
    events = []
    events_path = task_path / "events.jsonl"
    if events_path.exists():
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(
                        EventEntry(
                            event_type=event.get("event_type", "unknown"),
                            timestamp=event.get("timestamp", ""),
                            message=event.get("message", ""),
                            details=event.get("details"),
                        )
                    )
                except json.JSONDecodeError:
                    continue

    # Read explanations
    explanations: list[dict[str, Any]] = []
    explanations_path = task_path / "explanations.jsonl"
    if explanations_path.exists():
        with open(explanations_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    explanations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    title, summary, status = _parse_task_summary(task_path)

    # List artifact IDs
    artifact_ids = _list_artifact_ids(task_path)

    return InvestigationDetail(
        id=f"{swarm_id}/{task_id}",
        swarm_id=swarm_id,
        task_id=task_id,
        wandb_path=metadata.get("wandb_path") if metadata else None,
        created_at=_get_task_created_at(task_path, metadata),
        research_log=research_log,
        events=events,
        explanations=explanations,
        artifact_ids=artifact_ids,
        title=title,
        summary=summary,
        status=status,
    )


@router.get("/{swarm_id}/{task_id}/artifacts")
def list_artifacts(swarm_id: str, task_id: int) -> list[str]:
    """List all artifact IDs for an investigation."""
    task_path = SWARM_DIR / swarm_id / f"task_{task_id}"
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Investigation {swarm_id}/{task_id} not found")
    return _list_artifact_ids(task_path)


@router.get("/{swarm_id}/{task_id}/artifacts/{artifact_id}")
def get_artifact(swarm_id: str, task_id: int, artifact_id: str) -> dict[str, Any]:
    """Get a specific artifact by ID."""
    task_path = SWARM_DIR / swarm_id / f"task_{task_id}"
    artifact_path = task_path / "artifacts" / f"{artifact_id}.json"

    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact {artifact_id} not found in {swarm_id}/{task_id}",
        )

    data: dict[str, Any] = json.loads(artifact_path.read_text())
    return data
