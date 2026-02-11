"""Log scrapers for inferring job progress from SLURM log files.

Each scraper reads the tail of a log file and extracts a (current, total) progress
tuple. Returns None if no progress can be determined.

Log patterns by job type:
    harvest worker:  "[Worker 0] 81 batches"
    harvest merge:   tqdm "Merging worker states: 50%|...| 5/10"
                     then "Building components: 50%|...| 12/24"
    intruder eval:   "[N] scored M, $X.XX"  with header "Scoring T components"
    interpret:       "[N] $X.XX (...)"       with header "Interpreting T components"
    detection:       "[N] scored M, $X.XX"  with header "Scoring T components"
    fuzzing:         "[N] scored M, $X.XX"  with header "Scoring T components"
    attr worker:     tqdm "Attribution batches: 38%|...| 3841/10000"
    attr merge:      tqdm "Merging rank files: 50%|...| 63/127"
"""

import re
from pathlib import Path

from spd.settings import SLURM_LOGS_DIR


def _log_path(job_id: str, is_array: bool) -> Path:
    if is_array:
        return SLURM_LOGS_DIR / f"slurm-{job_id}_1.out"
    return SLURM_LOGS_DIR / f"slurm-{job_id}.out"


def _read_log_full(job_id: str, is_array: bool) -> str:
    """Read an entire log file."""
    path = _log_path(job_id, is_array)
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _read_log_tail(job_id: str, is_array: bool, n_lines: int = 200) -> str:
    """Read the last n_lines of a job's log file.

    For array jobs, reads worker 1's log (representative sample).
    """
    text = _read_log_full(job_id, is_array)
    if not text:
        return ""
    lines = text.split("\n")
    return "\n".join(lines[-n_lines:])


# tqdm outputs like "Harvesting:  38%|███▊      | 3841/10000 [06:04<09:27]"
# or on a single line with \r overwrites — the last occurrence wins.
_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")

# "[Worker 0] 81 batches"
_WORKER_BATCH_RE = re.compile(r"\[Worker \d+\] (\d+) batches")

# "[N] $X.XX (M in, K out)"  — interpret progress
_INTERPRET_PROGRESS_RE = re.compile(r"^\[(\d+)\] \$")

# "Interpreting N components" or "Scoring N components"
_TOTAL_COMPONENTS_RE = re.compile(r"(?:Interpreting|Scoring) (\d+) components")

# "[N] scored M, $X.XX" — scoring progress (intruder, detection, fuzzing)
_SCORED_RE = re.compile(r"^\[(\d+)\] scored (\d+),")

# "Processing complete. N batches" — harvest worker done
_HARVEST_COMPLETE_RE = re.compile(r"Processing complete\. (\d+) batches")


def scrape_harvest_worker(job_id: str, n_batches: int | None) -> tuple[int, int] | None:
    """Scrape harvest worker log for batch progress."""
    text = _read_log_tail(job_id, is_array=True)
    if not text:
        return None

    # Check for "Processing complete" first
    m = _HARVEST_COMPLETE_RE.search(text)
    if m:
        done = int(m.group(1))
        return (done, done) if n_batches is None else (done, n_batches)

    # Find last "[Worker N] M batches" line
    matches = _WORKER_BATCH_RE.findall(text)
    if matches:
        current = int(matches[-1])
        if n_batches is not None:
            return (current, n_batches)
        return (current, 0)

    return None


def scrape_tqdm(job_id: str, is_array: bool) -> tuple[int, int] | None:
    """Scrape tqdm progress from a log file. Returns last (current, total) found."""
    text = _read_log_tail(job_id, is_array=is_array)
    if not text:
        return None

    matches = _TQDM_RE.findall(text)
    if matches:
        current, total = int(matches[-1][0]), int(matches[-1][1])
        return (current, total)
    return None


def scrape_interpret(job_id: str) -> tuple[int, int] | None:
    """Scrape interpret job log for component progress.

    Reads the full log because progress lines ("[N] $X.XX") are sparse
    among retry/error noise.
    """
    text = _read_log_full(job_id, is_array=False)
    if not text:
        return None

    total_match = _TOTAL_COMPONENTS_RE.search(text)
    total = int(total_match.group(1)) if total_match else 0

    last_index = 0
    for line in text.split("\n"):
        m = _INTERPRET_PROGRESS_RE.match(line)
        if m:
            last_index = max(last_index, int(m.group(1)))

    if last_index > 0 or total > 0:
        return (last_index, total)
    return None


def scrape_scoring(job_id: str) -> tuple[int, int] | None:
    """Scrape scoring job log (intruder, detection, fuzzing) for progress.

    Reads the full log because "[N] scored M" lines are sparse among
    retry/error noise (~7 progress lines in 5000+ lines of retries).
    """
    text = _read_log_full(job_id, is_array=False)
    if not text:
        return None

    total_match = _TOTAL_COMPONENTS_RE.search(text)
    total = int(total_match.group(1)) if total_match else 0

    last_scored = 0
    for line in text.split("\n"):
        m = _SCORED_RE.match(line)
        if m:
            last_scored = max(last_scored, int(m.group(2)))

    if last_scored > 0 or total > 0:
        return (last_scored, total)
    return None


def scrape_progress(
    manifest_key: str, job_id: str, config: dict[str, object]
) -> tuple[int, int] | None:
    """Dispatch to the appropriate scraper based on manifest key.

    Returns (current, total) or None if no progress info found.
    """
    match manifest_key:
        case "harvest_array":
            harvest_cfg = config.get("harvest", {})
            inner_cfg = harvest_cfg.get("config", {}) if isinstance(harvest_cfg, dict) else {}
            n_batches = inner_cfg.get("n_batches") if isinstance(inner_cfg, dict) else None
            return scrape_harvest_worker(job_id, n_batches)

        case "harvest_merge":
            return scrape_tqdm(job_id, is_array=False)

        case "intruder_eval":
            return scrape_scoring(job_id)

        case "interpret":
            return scrape_interpret(job_id)

        case "detection" | "fuzzing":
            return scrape_scoring(job_id)

        case "attr_array":
            return scrape_tqdm(job_id, is_array=True)

        case "attr_merge":
            return scrape_tqdm(job_id, is_array=False)

        case _:
            return None
