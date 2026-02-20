"""Real-time progress monitoring for batch jobs.

Usage:
    # One-time status check
    python -m slurm.monitor outputs/jobs/job_20250112_143052

    # Watch mode (updates every 10s)
    python -m slurm.monitor outputs/jobs/job_20250112_143052 --watch

    # Custom refresh interval
    python -m slurm.monitor outputs/jobs/job_20250112_143052 --watch --interval 5
"""

import argparse
import sys
import time
from pathlib import Path

from .manifest import JobManifest


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def progress_bar(completed: int, total: int, width: int = 40) -> str:
    """Generate a text progress bar."""
    if total == 0:
        return "[" + " " * width + "]"

    filled = int(width * completed / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def estimate_eta(completed: int, total: int, avg_duration: float) -> str | None:
    """Estimate time remaining."""
    if completed == 0 or avg_duration == 0:
        return None

    remaining = total - completed
    eta_seconds = remaining * avg_duration
    return format_duration(eta_seconds)


def print_status(manifest: JobManifest, clear: bool = False):
    """Print current job status."""
    if clear:
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")

    progress = manifest.get_progress()

    job_id = progress["job_id"]
    total = progress["total"]
    completed = progress["completed"]
    failed = progress["failed"]
    running = progress["running"]
    pending = progress["pending"]
    avg_duration = progress["avg_duration_seconds"]

    # Header
    print(f"Job: {job_id}")
    print("=" * 60)

    # Progress bar
    bar = progress_bar(completed, total)
    percent = progress["percent_complete"]
    print(f"Progress: {bar} {percent:.1f}%")
    print()

    # Stats
    print(f"  Completed: {completed:>6}")
    print(f"  Failed:    {failed:>6}")
    print(f"  Running:   {running:>6}")
    print(f"  Pending:   {pending:>6}")
    print("  ─────────────────")
    print(f"  Total:     {total:>6}")
    print()

    # Timing
    if avg_duration > 0:
        print(f"Avg time per prompt: {format_duration(avg_duration)}")

    eta = estimate_eta(completed, total, avg_duration)
    if eta:
        print(f"Estimated time remaining: {eta}")

    # Status
    if completed == total:
        print("\n✓ Job complete!")
        if failed > 0:
            print(f"  ({failed} failed prompts)")
    elif running > 0:
        print(f"\n⟳ Job running ({running} active tasks)")
    elif pending > 0:
        print("\n⏸ Job pending")

    # Show recent failures if any
    if failed > 0:
        print("\nRecent failures:")
        failed_prompts = manifest.get_failed_prompts()[:5]
        for fp in failed_prompts:
            prompt_preview = fp.prompt[:40] + "..." if len(fp.prompt) > 40 else fp.prompt
            error_preview = (fp.error or "Unknown error")[:50]
            print(f"  [{fp.idx}] {prompt_preview}")
            print(f"       Error: {error_preview}")


def watch_job(manifest: JobManifest, interval: int = 10):
    """Watch job progress with periodic updates."""
    try:
        while True:
            print_status(manifest, clear=True)

            progress = manifest.get_progress()
            if progress["completed"] + progress["failed"] == progress["total"]:
                break

            print(f"\nRefreshing every {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor batch job progress"
    )
    parser.add_argument(
        "job_dir",
        type=Path,
        help="Path to job directory",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode (continuous updates)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)",
    )

    args = parser.parse_args()

    manifest = JobManifest(args.job_dir)
    if not manifest.exists():
        print(f"Error: No manifest found in {args.job_dir}", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        watch_job(manifest, args.interval)
    else:
        print_status(manifest)


if __name__ == "__main__":
    main()
