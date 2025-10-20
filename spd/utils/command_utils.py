"""Minimal utilities for running shell-safe commands locally."""

import shlex
import subprocess

from spd.log import logger


def run_script_array_local(commands: list[str], parallel: bool = False) -> None:
    """Run multiple shell-safe command strings locally.

    Args:
        commands: List of shell-safe command strings (built with shlex.join())
        parallel: If True, run all commands in parallel. If False, run sequentially.
    """
    n_commands = len(commands)

    if not parallel:
        logger.section(f"LOCAL EXECUTION: Running {n_commands} tasks serially")
        for i, cmd in enumerate(commands, 1):
            logger.info(f"[{i}/{n_commands}] Running: {cmd}")
            subprocess.run(shlex.split(cmd), shell=False, check=True)
        logger.section("LOCAL EXECUTION COMPLETE")
    else:
        logger.section(f"LOCAL EXECUTION: Starting {n_commands} tasks in parallel")
        procs: list[subprocess.Popen[bytes]] = []
        for i, cmd in enumerate(commands, 1):
            logger.info(f"[{i}/{n_commands}] Starting: {cmd}")
            proc = subprocess.Popen(shlex.split(cmd), shell=False)
            procs.append(proc)

        logger.section("WAITING FOR ALL TASKS TO COMPLETE")
        for proc in procs:
            proc.wait()
            if proc.returncode != 0:
                logger.error(f"Process {proc.pid} failed with exit code {proc.returncode}")
        logger.section("LOCAL EXECUTION COMPLETE")
