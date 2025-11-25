"""Simple timing utility that writes to a shared JSONL file."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

TIMING_FILE = Path(__file__).parent.parent.parent / "timing.jsonl"


def log_timing(event: str, duration_ms: float, **extra: str | int | float) -> None:
    """Log a timing event to the shared timing file."""
    entry = {
        "ts": time.time(),
        "source": "backend",
        "event": event,
        "duration_ms": round(duration_ms, 2),
        **extra,
    }
    with open(TIMING_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


@contextmanager
def timed(event: str, **extra: str | int | float):
    """Context manager to time a block and log it."""
    start = time.perf_counter()
    yield
    duration_ms = (time.perf_counter() - start) * 1000
    log_timing(event, duration_ms, **extra)


def clear_timing_file() -> None:
    """Clear the timing file."""
    TIMING_FILE.write_text("")
