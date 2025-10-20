"""Minimal utilities for running shell-safe commands locally."""

import subprocess
import tempfile
from pathlib import Path

from spd.log import logger


def run_script_array_local(
    commands: list[str], parallel: bool = False, track_resources: bool = False
) -> dict[str, dict[str, float]] | None:
    """Run multiple shell-safe command strings locally.

    Args:
        commands: List of shell-safe command strings (built with shlex.join())
        parallel: If True, run all commands in parallel. If False, run sequentially.
        track_resources: If True, track and return resource usage for each command using /usr/bin/time.

    Returns:
        If track_resources is True, returns dict mapping commands to resource metrics dict.
        Resource metrics include: K (avg memory KB), M (max memory KB), P (CPU %),
        S (system CPU sec), U (user CPU sec), e (wall time sec).
        Otherwise returns None.
    """
    n_commands = len(commands)
    resources: dict[str, dict[str, float]] = {}
    resource_files: list[Path] = []

    # Wrap commands with /usr/bin/time if resource tracking is requested
    if track_resources:
        wrapped_commands: list[str] = []
        for cmd in commands:
            resource_file = Path(tempfile.mktemp(suffix=".resources"))  # pyright: ignore[reportDeprecated]
            resource_files.append(resource_file)
            # Use /usr/bin/time to track comprehensive resource usage
            # K=avg total mem, M=max resident, P=CPU%, S=system time, U=user time, e=wall time
            wrapped_cmd = f'/usr/bin/time -f "K:%K M:%M P:%P S:%S U:%U e:%e" -o {resource_file} {cmd}'
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

        # Read resource usage results
        if track_resources:
            for cmd, resource_file in zip(commands, resource_files, strict=True):
                if resource_file.exists():
                    # Parse format: "K:123 M:456 P:78% S:1.23 U:4.56 e:7.89"
                    output = resource_file.read_text().strip()
                    metrics: dict[str, float] = {}

                    for part in output.split():
                        if ":" in part:
                            key, value = part.split(":", 1)
                            # Remove % sign from CPU percentage
                            value = value.rstrip("%")
                            try:
                                metrics[key] = float(value)
                            except ValueError:
                                logger.warning(f"Could not parse {key}:{value} for command: {cmd}")

                    resources[cmd] = metrics
                else:
                    logger.warning(f"Resource file not found for: {cmd}")

            # Log comprehensive resource usage table
            logger.section("RESOURCE USAGE RESULTS")
            for cmd, metrics in resources.items():
                logger.info(f"Command: {cmd}")
                logger.info(
                    f"  Time: {metrics.get('e', 0):.2f}s wall, "
                    f"{metrics.get('U', 0):.2f}s user, "
                    f"{metrics.get('S', 0):.2f}s system"
                )
                logger.info(
                    f"  Memory: {metrics.get('M', 0) / 1024:.1f} MB peak, "
                    f"{metrics.get('K', 0) / 1024:.1f} MB avg"
                )
                logger.info(f"  CPU: {metrics.get('P', 0):.1f}%")

    finally:
        # Clean up temp files
        if track_resources:
            for resource_file in resource_files:
                if resource_file.exists():
                    resource_file.unlink()

    return resources if track_resources else None
