"""Minimal utilities for running shell-safe commands locally."""

import subprocess
import tempfile
from pathlib import Path

from spd.log import logger


def run_script_array_local(
    commands: list[str], parallel: bool = False, track_timing: bool = False
) -> dict[str, float] | None:
    """Run multiple shell-safe command strings locally.

    Args:
        commands: List of shell-safe command strings (built with shlex.join())
        parallel: If True, run all commands in parallel. If False, run sequentially.
        track_timing: If True, track and return timing for each command using /usr/bin/time.

    Returns:
        If track_timing is True, returns dict mapping commands to execution times in seconds.
        Otherwise returns None.
    """
    n_commands = len(commands)
    timings: dict[str, float] = {}
    time_files: list[Path] = []

    # Wrap commands with /usr/bin/time if timing is requested
    if track_timing:
        wrapped_commands: list[str] = []
        for cmd in commands:
            time_file = Path(tempfile.mktemp(suffix=".time"))
            time_files.append(time_file)
            # Use /usr/bin/time to track wall-clock time
            wrapped_cmd = f'/usr/bin/time -f "%e" -o {time_file} {cmd}'
            wrapped_commands.append(wrapped_cmd)
        commands_to_run = wrapped_commands
    else:
        commands_to_run = commands

    try:
        if not parallel:
            logger.section(f"LOCAL EXECUTION: Running {n_commands} tasks serially")
            for i, cmd in enumerate(commands_to_run, 1):
                logger.info(f"[{i}/{n_commands}] Running: {commands[i - 1]}")
                subprocess.run(cmd, shell=True, check=True)
            logger.section("LOCAL EXECUTION COMPLETE")
        else:
            logger.section(f"LOCAL EXECUTION: Starting {n_commands} tasks in parallel")
            procs: list[subprocess.Popen[bytes]] = []

            for i, cmd in enumerate(commands_to_run, 1):
                logger.info(f"[{i}/{n_commands}] Starting: {commands[i - 1]}")
                proc = subprocess.Popen(cmd, shell=True)
                procs.append(proc)

            logger.section("WAITING FOR ALL TASKS TO COMPLETE")
            for proc, cmd in zip(procs, commands, strict=True):  # noqa: B007
                proc.wait()
                if proc.returncode != 0:
                    logger.error(f"Process {proc.pid} failed with exit code {proc.returncode}")
            logger.section("LOCAL EXECUTION COMPLETE")

        # Read timing results
        if track_timing:
            for cmd, time_file in zip(commands, time_files, strict=True):
                if time_file.exists():
                    elapsed = float(time_file.read_text().strip())
                    timings[cmd] = elapsed
                else:
                    logger.warning(f"Timing file not found for: {cmd}")

            logger.section("TIMING RESULTS")
            for cmd, elapsed in timings.items():
                logger.info(f"{elapsed:.2f}s - {cmd}")

    finally:
        # Clean up temp files
        if track_timing:
            for time_file in time_files:
                if time_file.exists():
                    time_file.unlink()

    return timings if track_timing else None
