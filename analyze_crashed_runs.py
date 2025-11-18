#!/usr/bin/env python3
"""
Analyze crashed W&B runs and correlate with SLURM hosts.

This script:
1. Fetches all crashed W&B runs from the last 10 days (with step > 1)
2. Searches SLURM logs (oli + lucius) to find which job ran each W&B run
3. Uses sacct to map SLURM job IDs to host names
4. Generates a summary of crashes by host
"""

import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import wandb


def get_crashed_runs(project: str, entity: str, days: int = 10) -> list[dict]:
    """Fetch crashed W&B runs from the last N days with step > 1."""
    api = wandb.Api(timeout=60)

    cutoff_date = datetime.now() - timedelta(days=days)

    print(f"Fetching crashed/failed runs from W&B project: {entity}/{project}")
    print(f"Looking for runs after: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get all runs, filter in Python since W&B API filtering can be limited
    runs = api.runs(f"{entity}/{project}")

    crashed_runs = []
    for run in runs:
        # Check if run crashed or failed
        if run.state not in ("crashed", "failed"):
            continue

        # Check if created in last N days
        # Handle both formats: with and without 'Z' suffix
        created_at_str = run.created_at.rstrip("Z")
        created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%S")
        if created_at < cutoff_date:
            continue

        # Check if step > 0 (meaning it actually started running)
        try:
            summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else run.summary
            if isinstance(summary, str):
                # Skip if summary is a string (malformed)
                continue
            step = summary.get("_step", 0) if isinstance(summary, dict) else 0
            if step < 1:
                # Skip runs that never got to step 1
                continue
        except (AttributeError, TypeError):
            # Skip runs with malformed summary
            continue

        crashed_runs.append({
            "id": run.id,
            "name": run.name,
            "created_at": created_at,
            "state": run.state,
            "step": step,
            "url": run.url,
        })

    print(f"Found {len(crashed_runs)} crashed runs with step > 1")
    return crashed_runs


def find_slurm_log_with_wandb_id(wandb_id: str, log_dirs: list[Path]) -> Path | None:
    """Search SLURM logs for a specific W&B run ID."""
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue

        # Find all .out files in the directory
        log_files = list(log_dir.glob("slurm-*.out"))

        for log_file in log_files:
            try:
                # Use grep to search for the W&B ID (much faster than reading in Python)
                result = subprocess.run(
                    ["grep", "-l", wandb_id, str(log_file)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return log_file
            except (subprocess.TimeoutExpired, Exception):
                continue

    return None


def extract_job_id_from_log_path(log_path: Path) -> str | None:
    """Extract SLURM job ID from log filename (e.g., slurm-28170_1.out -> 28170)."""
    match = re.match(r"slurm-(\d+)(?:_\d+)?\.out", log_path.name)
    if match:
        return match.group(1)
    return None


def get_host_from_job_id(job_id: str, start_date: str = "2025-11-07") -> str | None:
    """Get host name from SLURM job ID using sacct."""
    try:
        result = subprocess.run(
            ["sacct", "--format=JobID,NodeList%50,State,Start", "-S", start_date, "-j", job_id],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if len(lines) < 3:  # Header + separator + at least one data line
            return None

        # Look for the main job line (not .batch or .extern)
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            if ".batch" not in parts[0] and ".extern" not in parts[0] and "." not in parts[0]:
                return parts[1]

        # Fallback: return first non-header line's host
        parts = lines[2].split()
        if len(parts) >= 2:
            return parts[1]

    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Error getting host for job {job_id}: {e}")

    return None


def main():
    # Configuration
    PROJECT = "spd"
    ENTITY = "goodfire"
    DAYS = 14

    # SLURM log directories
    OLI_LOGS = Path.home() / "slurm_logs"
    LUCIUS_LOGS = Path("/mnt/polished-lake/home/lucius/slurm_logs")

    LOG_DIRS = [OLI_LOGS, LUCIUS_LOGS]

    print("=" * 80)
    print("CRASHED W&B RUNS ANALYSIS")
    print("=" * 80)
    print()

    # Step 1: Get crashed runs from W&B
    crashed_runs = get_crashed_runs(PROJECT, ENTITY, DAYS)

    if not crashed_runs:
        print("No crashed runs found!")
        return

    print()
    print("=" * 80)
    print("REVERSE LOOKUP: W&B ID -> SLURM JOB ID -> HOST")
    print("=" * 80)
    print()

    # Step 2: Find SLURM logs and map to hosts
    results = []
    not_found = []

    for i, run in enumerate(crashed_runs, 1):
        wandb_id = run["id"]
        print(f"[{i}/{len(crashed_runs)}] Processing W&B run: {wandb_id} ({run['name']})")

        # Find SLURM log containing this W&B ID
        log_path = find_slurm_log_with_wandb_id(wandb_id, LOG_DIRS)

        if not log_path:
            print(f"  ⚠️  SLURM log not found")
            not_found.append(run)
            continue

        # Extract SLURM job ID
        job_id = extract_job_id_from_log_path(log_path)
        if not job_id:
            print(f"  ⚠️  Could not extract job ID from: {log_path.name}")
            not_found.append(run)
            continue

        # Get host from sacct
        host = get_host_from_job_id(job_id)
        if not host:
            print(f"  ⚠️  Could not get host for job ID: {job_id}")
            not_found.append(run)
            continue

        print(f"  ✓ SLURM job: {job_id}, Host: {host}, Log: {log_path.name}")

        results.append({
            "wandb_id": wandb_id,
            "wandb_name": run["name"],
            "wandb_url": run["url"],
            "created_at": run["created_at"],
            "step": run["step"],
            "slurm_job_id": job_id,
            "slurm_log": log_path.name,
            "host": host,
        })

    print()
    print("=" * 80)
    print("SUMMARY BY HOST")
    print("=" * 80)
    print()

    # Step 3: Summarize by host
    host_crashes = defaultdict(list)
    for result in results:
        host_crashes[result["host"]].append(result)

    # Sort hosts by number of crashes
    sorted_hosts = sorted(host_crashes.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"Total crashed runs analyzed: {len(crashed_runs)}")
    print(f"Successfully mapped to hosts: {len(results)}")
    print(f"Not found in logs: {len(not_found)}")
    print(f"Unique hosts with crashes: {len(host_crashes)}")
    print()

    for host, crashes in sorted_hosts:
        print(f"{host}: {len(crashes)} crashes")
        for crash in crashes[:5]:  # Show first 5
            print(f"  - Job {crash['slurm_job_id']}: {crash['wandb_name']} (step {crash['step']})")
        if len(crashes) > 5:
            print(f"  ... and {len(crashes) - 5} more")
        print()

    # Step 4: Save detailed results
    output_file = Path("crashed_runs_by_host.txt")
    with open(output_file, "w") as f:
        f.write("CRASHED W&B RUNS - DETAILED REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Time range: Last {DAYS} days\n")
        f.write(f"Total crashed runs: {len(crashed_runs)}\n")
        f.write(f"Successfully mapped: {len(results)}\n")
        f.write(f"Not found: {len(not_found)}\n\n")

        f.write("SUMMARY BY HOST\n")
        f.write("=" * 80 + "\n\n")

        for host, crashes in sorted_hosts:
            f.write(f"\n{host}: {len(crashes)} crashes\n")
            f.write("-" * 80 + "\n")
            for crash in crashes:
                f.write(f"  SLURM Job: {crash['slurm_job_id']}\n")
                f.write(f"  W&B Run:   {crash['wandb_name']}\n")
                f.write(f"  W&B ID:    {crash['wandb_id']}\n")
                f.write(f"  Created:   {crash['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Step:      {crash['step']}\n")
                f.write(f"  URL:       {crash['wandb_url']}\n")
                f.write(f"  Log:       {crash['slurm_log']}\n")
                f.write("\n")

        if not_found:
            f.write("\n\nNOT FOUND IN LOGS\n")
            f.write("=" * 80 + "\n\n")
            for run in not_found:
                f.write(f"  W&B ID:   {run['id']}\n")
                f.write(f"  Name:     {run['name']}\n")
                f.write(f"  Created:  {run['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Step:     {run['step']}\n")
                f.write("\n")

    print(f"Detailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
