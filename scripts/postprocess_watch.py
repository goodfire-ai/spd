#!/usr/bin/env python3
"""Watch-style monitor for spd-postprocess SLURM jobs.

Shows a live-updating dependency graph with progress bars for array jobs,
color-coded states, and elapsed times.

Usage:
    python scripts/postprocess_watch.py                  # auto-detect recent jobs
    python scripts/postprocess_watch.py --interval 5     # custom refresh interval
    python scripts/postprocess_watch.py --once            # print once and exit
"""

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime

# ── ANSI ─────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
WHITE = "\033[97m"

# ── Pipeline topology ────────────────────────────────────────────────────────
#
# Each stage: (display_label, name_matcher, is_array)
# name_matcher: str -> bool, matching sacct JobName


def _harvest_array(n: str) -> bool:
    return n.startswith("spd-harvest") and n != "spd-harvest-merge"


def _attr_array(n: str) -> bool:
    return n.startswith("spd-attr") and n != "spd-attr-merge"


STAGES: list[tuple[str, list[tuple[str, ..., bool]]]] = [
    (
        "HARVEST",
        [
            ("harvest", _harvest_array, True),
            ("merge", lambda n: n == "spd-harvest-merge", False),
            ("intruder", lambda n: n == "spd-intruder-eval", False),
        ],
    ),
    (
        "AUTOINTERP",
        [
            ("interpret", lambda n: n == "spd-interpret", False),
            ("detection", lambda n: n == "spd-detection", False),
            ("fuzzing", lambda n: n == "spd-fuzzing", False),
        ],
    ),
    (
        "ATTRIBUTIONS",
        [
            ("attr", _attr_array, True),
            ("merge", lambda n: n == "spd-attr-merge", False),
        ],
    ),
]

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class JobInfo:
    job_id: str
    name: str
    state: str
    elapsed: str
    exit_code: str
    n_tasks: int = 0
    tasks_completed: int = 0
    tasks_running: int = 0
    tasks_failed: int = 0
    tasks_pending: int = 0


def query_jobs() -> dict[str, JobInfo]:
    """Query sacct for recent spd-* jobs, return latest job per stage."""
    cmd = [
        "sacct",
        "--format=JobID,JobName%30,State%20,Elapsed,ExitCode",
        "--user",
        os.environ.get("USER", ""),
        "--starttime",
        "now-48hours",
        "-n",
        "--parsable2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}

    parents: dict[str, dict] = {}
    array_tasks: dict[str, list[str]] = {}

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        raw_id, name, state, elapsed, exit_code = (p.strip() for p in parts[:5])

        # Normalize "CANCELLED by 12345" -> "CANCELLED"
        state = state.split()[0]

        if not name.startswith("spd-"):
            continue

        # Array task: "12345_1" (skip ".batch"/".extern" sub-steps)
        m = re.match(r"^(\d+)_(\d+)$", raw_id)
        if m:
            array_tasks.setdefault(m.group(1), []).append(state)
            continue

        # Sub-step like "12345.batch" — skip
        if "." in raw_id:
            continue

        parents[raw_id] = {
            "job_id": raw_id,
            "name": name,
            "state": state,
            "elapsed": elapsed,
            "exit_code": exit_code,
        }

    # Keep the most recent (highest job_id) per stage
    result_map: dict[str, JobInfo] = {}

    for group_name, stages in STAGES:
        for label, matcher, _is_array in stages:
            key = f"{group_name}:{label}"
            best_id = -1
            best_info = None

            for job_id, raw in parents.items():
                if not matcher(raw["name"]):
                    continue
                jid = int(job_id)
                if jid > best_id:
                    best_id = jid
                    info = JobInfo(**raw)
                    if job_id in array_tasks:
                        tasks = array_tasks[job_id]
                        info.n_tasks = len(tasks)
                        info.tasks_completed = sum(1 for t in tasks if t == "COMPLETED")
                        info.tasks_running = sum(1 for t in tasks if t == "RUNNING")
                        info.tasks_failed = sum(
                            1 for t in tasks if t in TERMINAL_STATES and t != "COMPLETED"
                        )
                        info.tasks_pending = sum(1 for t in tasks if t == "PENDING")
                    best_info = info

            if best_info is not None:
                result_map[key] = best_info

    return result_map


# ── Rendering ────────────────────────────────────────────────────────────────

BAR_W = 20
W = 70


def _icon(state: str) -> tuple[str, str]:
    """(color, icon) for a state."""
    match state:
        case "COMPLETED":
            return GREEN, "●"
        case "RUNNING":
            return YELLOW, "◑"
        case "PENDING":
            return DIM, "○"
        case _:
            return RED, "✗"


def _bar(done: int, running: int, failed: int, total: int) -> str:
    if total == 0:
        return DIM + "░" * BAR_W + RESET
    w_d = round(done / total * BAR_W)
    w_r = round(running / total * BAR_W)
    w_f = round(failed / total * BAR_W)
    w_p = BAR_W - w_d - w_r - w_f
    return GREEN + "█" * w_d + YELLOW + "█" * w_r + RED + "█" * w_f + DIM + "░" * w_p + RESET


def _elapsed(e: str) -> str:
    """'00:03:12' -> '3:12', '02:15:30' -> '2:15:30'."""
    if not e or e == "00:00:00":
        return ""
    if "-" in e:
        return e
    h, m, s = e.split(":")
    return f"{int(m)}:{s}" if int(h) == 0 else f"{int(h)}:{m}:{s}"


def _render_array(label: str, job: JobInfo) -> str:
    color, ic = _icon(job.state)
    bar = _bar(job.tasks_completed, job.tasks_running, job.tasks_failed, job.n_tasks)
    frac = f"{job.tasks_completed}/{job.n_tasks}"
    el = _elapsed(job.elapsed)
    fail = f"  {RED}{job.tasks_failed} failed{RESET}" if job.tasks_failed else ""
    jid = f"{DIM}#{job.job_id}{RESET}"
    return (
        f"    {color}{ic}{RESET} {label:<12} {bar}  {frac:>5}  "
        f"{color}{job.state:<12}{RESET} {DIM}{el:>8}{RESET}  {jid}{fail}"
    )


def _render_single(label: str, job: JobInfo) -> str:
    color, ic = _icon(job.state)
    el = _elapsed(job.elapsed)
    jid = f"{DIM}#{job.job_id}{RESET}"
    match job.state:
        case "COMPLETED":
            bar = f"{GREEN}{'━' * BAR_W}{RESET}"
        case "RUNNING":
            h = BAR_W // 2
            bar = f"{YELLOW}{'━' * h}{'╌' * (BAR_W - h)}{RESET}"
        case "PENDING":
            bar = f"{DIM}{'· ' * (BAR_W // 2)}{RESET}"
        case _:
            bar = f"{RED}{'━' * BAR_W}{RESET}"
    return (
        f"    {color}{ic}{RESET} {label:<12} {bar}{'':>7}"
        f"{color}{job.state:<12}{RESET} {DIM}{el:>8}{RESET}  {jid}"
    )


def _render_missing(label: str) -> str:
    return f"    {DIM}  {label:<12} {'· ' * (BAR_W // 2):<{BAR_W}}{'':>7}{'—':>12}{RESET}"


def render(jobs: dict[str, JobInfo]) -> str:
    lines: list[str] = []
    now = datetime.now().strftime("%H:%M:%S")

    # Header
    lines.append("\033[2J\033[H")
    lines.append("")
    title = "spd-postprocess"
    pad = W - len(title) - len(now) - 4
    lines.append(f"  {BOLD}{WHITE}{title}{RESET}{' ' * pad}{DIM}{now}{RESET}")
    lines.append(f"  {DIM}{'━' * (W - 2)}{RESET}")

    n_total = 0
    n_done = 0
    has_fail = False

    for group_name, stages in STAGES:
        group_keys = [f"{group_name}:{label}" for label, _, _ in stages]
        group_jobs = [jobs.get(k) for k in group_keys]

        if not any(j is not None for j in group_jobs):
            continue

        lines.append("")
        lines.append(f"  {BOLD}{CYAN}{group_name}{RESET}")

        for (label, _, is_array), job in zip(stages, group_jobs):
            n_total += 1

            if job is None:
                lines.append(_render_missing(label))
                continue

            if job.state == "COMPLETED":
                n_done += 1
            if job.state in TERMINAL_STATES and job.state != "COMPLETED":
                has_fail = True

            if is_array and job.n_tasks > 0:
                lines.append(_render_array(label, job))
            else:
                lines.append(_render_single(label, job))

        # Dependency arrow
        if group_name == "HARVEST":
            harvest_merge = jobs.get("HARVEST:merge")
            if harvest_merge is None or harvest_merge.state != "COMPLETED":
                lines.append(f"    {DIM}╰───▷ autointerp waits on harvest merge{RESET}")

    # Footer
    lines.append("")
    lines.append(f"  {DIM}{'━' * (W - 2)}{RESET}")

    legend = f"  {GREEN}●{RESET} done  {YELLOW}◑{RESET} running  {DIM}○{RESET} pending  {RED}✗{RESET} failed"
    if n_total > 0:
        frac = f"{n_done}/{n_total}"
        if has_fail:
            styled = f"{RED}{frac}{RESET}"
        elif n_done == n_total:
            styled = f"{GREEN}{frac} ✓{RESET}"
        else:
            styled = f"{DIM}{frac}{RESET}"
        lines.append(f"{legend}{'':>{W - 50}}{styled}")
    else:
        lines.append(f"  {DIM}No spd-postprocess jobs found in the last 48h.{RESET}")
        lines.append(f"  {DIM}Waiting... (Ctrl-C to quit){RESET}")

    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Watch spd-postprocess SLURM jobs")
    parser.add_argument("--interval", "-n", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    try:
        while True:
            jobs = query_jobs()
            print(render(jobs), flush=True)

            if args.once:
                break

            # Auto-exit when everything is terminal
            if jobs and all(j.state in TERMINAL_STATES for j in jobs.values()):
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
