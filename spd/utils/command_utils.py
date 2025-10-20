"""Minimal utilities for running shell-safe commands locally."""

import shlex
import subprocess

from spd.log import logger


def run_script_array_local(
    commands: list[str], parallel: bool = False, track_timing: bool = False
) -> dict[str, float] | None:
    """Run multiple shell-safe command strings locally.

    Args:
        commands: List of shell-safe command strings (built with shlex.join())
        parallel: If True, run all commands in parallel. If False, run sequentially.
        track_timing: If True, track and return timing for each command.

    Returns:
        If track_timing is True, returns dict mapping commands to execution times in seconds.
        Otherwise returns None.
    """
    import time

    n_commands = len(commands)
    timings: dict[str, float] = {} if track_timing else {}

    if not parallel:
        logger.section(f"LOCAL EXECUTION: Running {n_commands} tasks serially")
        for i, cmd in enumerate(commands, 1):
            logger.info(f"[{i}/{n_commands}] Running: {cmd}")
            start_time = time.time() if track_timing else 0.0
            subprocess.run(shlex.split(cmd), shell=False, check=True)
            if track_timing:
                elapsed = time.time() - start_time
                timings[cmd] = elapsed
        logger.section("LOCAL EXECUTION COMPLETE")
    else:
        logger.section(f"LOCAL EXECUTION: Starting {n_commands} tasks in parallel")
        procs: list[subprocess.Popen[bytes]] = []
        start_times: list[float] = []

        for i, cmd in enumerate(commands, 1):
            logger.info(f"[{i}/{n_commands}] Starting: {cmd}")
            start_time = time.time() if track_timing else 0.0
            proc = subprocess.Popen(shlex.split(cmd), shell=False)
            procs.append(proc)
            start_times.append(start_time)

        logger.section("WAITING FOR ALL TASKS TO COMPLETE")
        for proc, start_time, cmd in zip(procs, start_times, commands, strict=True):
            proc.wait()
            if track_timing:
                elapsed = time.time() - start_time
                timings[cmd] = elapsed
            if proc.returncode != 0:
                logger.error(f"Process {proc.pid} failed with exit code {proc.returncode}")
        logger.section("LOCAL EXECUTION COMPLETE")

    if track_timing:
        logger.section("TIMING RESULTS")
        for cmd, elapsed in timings.items():
            logger.info(f"{elapsed:.2f}s - {cmd}")

    return timings if track_timing else None
