import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, override

from spd.log import logger


@dataclass
class Command:
    """Simple typed command with shell flag and subprocess helpers."""

    cmd: list[str] | str
    shell: bool = False
    env: dict[str, str] | None = None
    inherit_env: bool = True

    def __post_init__(self) -> None:
        """Enforce cmd type when shell is False."""
        if self.shell is False and isinstance(self.cmd, str):
            raise ValueError("cmd must be list[str] when shell is False")

    def _quote_env(self) -> str:
        """Return KEY=VAL tokens for env values. ignores `inherit_env`."""
        if not self.env:
            return ""

        parts: list[str] = []
        for k, v in self.env.items():
            token: str = f"{k}={v}"
            parts.append(token)
        prefix: str = " ".join(parts)
        return prefix

    @property
    def cmd_joined(self) -> str:
        """Return cmd as a single string, joining with spaces if it's a list. no env included."""
        if isinstance(self.cmd, str):
            return self.cmd
        else:
            return " ".join(self.cmd)

    @property
    def cmd_for_subprocess(self) -> list[str] | str:
        """Return cmd, splitting if shell is True and cmd is a string."""
        if self.shell:
            if isinstance(self.cmd, str):
                return self.cmd
            else:
                return " ".join(self.cmd)
        else:
            assert isinstance(self.cmd, list)
            return self.cmd

    def script_line(self) -> str:
        """Return a single shell string, prefixing KEY=VAL for env if provided."""
        return f"{self._quote_env()} {self.cmd_joined}".strip()

    @property
    def env_final(self) -> dict[str, str]:
        """Return final env dict, merging with os.environ if inherit_env is True."""
        return {
            **(os.environ if self.inherit_env else {}),
            **(self.env or {}),
        }

    def run(
        self,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        """Call subprocess.run with this command."""
        try:
            return subprocess.run(
                self.cmd_for_subprocess,
                shell=self.shell,
                env=self.env_final,
                **kwargs,
            )
        except subprocess.CalledProcessError as e:
            print(f"Command failed: `{self.script_line()}`", file=sys.stderr)
            raise e

    def Popen(
        self,
        **kwargs: Any,
    ) -> subprocess.Popen[Any]:
        """Call subprocess.Popen with this command."""
        return subprocess.Popen(
            self.cmd_for_subprocess,
            shell=self.shell,
            env=self.env_final,
            **kwargs,
        )

    @override
    def __str__(self) -> str:
        return self.script_line()


def run_script_array_local(commands: list[Command], parallel: bool = False) -> None:
    """alternative to `create_slurm_array_script` for local execution of multiple commands

    if `parallel` is True, runs all commands in parallel using subprocess.Popen
    otherwise runs them sequentially using subprocess.run
    """
    n_commands: int = len(commands)
    if not parallel:
        logger.section(f"LOCAL EXECUTION: Running {n_commands} tasks serially")
        for i, cmd in enumerate(commands):
            logger.info(f"[{i + 1}/{n_commands}] Running command: `{cmd.cmd_joined}`")
            cmd.run(check=True)
        logger.section("LOCAL EXECUTION COMPLETE")
    else:
        procs: list[subprocess.Popen[bytes]] = []
        for i, cmd in enumerate(commands):
            logger.info(f"[{i + 1}/{n_commands}] Starting command: `{cmd.cmd_joined}`")
            p = cmd.Popen()
            procs.append(p)
        logger.section("STARTED ALL COMMANDS")
        for p in procs:
            p.wait()
            logger.info(f"Process {p.pid} finished with exit code {p.returncode}")
        logger.section("LOCAL EXECUTION COMPLETE")
