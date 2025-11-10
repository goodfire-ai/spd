"""
Development server launcher for SPD app.
Starts backend and frontend with:
  - Automatic port detection (with --strictPort for Vite)
  - TCP-based health checks (no false negatives on 404)
  - Graceful shutdown of process groups
  - Clear logging & dependency checks
"""

import atexit
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import TextIO

import requests


class AnsiEsc(StrEnum):
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


APP_DIR = Path(__file__).parent.resolve()
LOGS_DIR = APP_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

STARTUP_TIMEOUT_SECONDS = 30
BACKEND_DEFAULT_START = 8000
FRONTEND_DEFAULT_START = 5173


def is_port_in_use(port: int) -> bool:
    """Best-effort check: try binding on loopback IPv4 and IPv6."""
    # IPv4
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s4:
        s4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s4.bind(("127.0.0.1", port))
        except OSError:
            return True

    # IPv6 (if available)
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s6:
            s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s6.bind(("::1", port))
            except OSError:
                return True
    except OSError:
        # IPv6 not supported; ignore
        pass

    return False


def find_available_port(start_port: int) -> int:
    """Find an available port in [start_port, start_port+100)."""
    for port in range(start_port, start_port + 100):
        if not is_port_in_use(port):
            return port
    print(
        f"{AnsiEsc.RED}✗{AnsiEsc.RESET} Could not find available port after checking 100 ports from {start_port}",
        file=sys.stderr,
    )
    sys.exit(1)


def _spawn(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    logfile: TextIO,
) -> subprocess.Popen[str]:
    """Spawn a process in its own process group, streaming stdout/stderr to logfile."""
    try:
        # Use preexec_fn to set process group on Unix systems
        # This allows us to kill the entire process tree later
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=logfile,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,  # modern alias for universal_newlines=True
            preexec_fn=os.setpgrp,  # Create new process group
            env=env,
        )
    except FileNotFoundError as e:
        print(
            f"{AnsiEsc.RED}✗ Failed to start:{AnsiEsc.RESET} {' '.join(cmd)}\n"
            f"{AnsiEsc.DIM}{e}{AnsiEsc.RESET}",
            file=sys.stderr,
        )
        sys.exit(1)


class AppRunner:
    """Manages backend and frontend processes with proper cleanup on signals."""

    def __init__(self):
        self.backend_process: subprocess.Popen[str] | None = None
        self.frontend_process: subprocess.Popen[str] | None = None
        self.cleanup_called = False

    def wait_to_serve(self, port: int) -> None:
        start = time.time()
        while time.time() < (start + STARTUP_TIMEOUT_SECONDS):
            try:
                response = requests.get(f"http://localhost:{port}", timeout=1.0)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        print(f"{AnsiEsc.RED}✗{AnsiEsc.RESET} Healthcheck failed", file=sys.stderr)
        sys.exit(1)

    def start_backend(self, port: int, logfile: TextIO) -> None:
        print(f"  {AnsiEsc.DIM}▸ Starting backend...{AnsiEsc.RESET}", end="", flush=True)
        project_root = APP_DIR.parent.parent
        cmd = [
            "uv",
            "run",
            "python",
            "-u",
            str(APP_DIR / "backend" / "server.py"),
            "--port",
            str(port),
        ]
        proc = _spawn(cmd, cwd=project_root, env=None, logfile=logfile)
        self.backend_process = proc  # Immediately visible to signal handler
        time.sleep(0.5)
        print(
            f"\r  {AnsiEsc.DIM}▸ Waiting for backend to serve...{AnsiEsc.RESET}", end="", flush=True
        )
        self.wait_to_serve(port)
        print(
            f"\r  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} Backend started (pid {proc.pid}, port {port})        "
        )

    def start_frontend(self, port: int, backend_port: int, logfile: TextIO) -> None:
        print(f"  {AnsiEsc.DIM}▸ Starting frontend...{AnsiEsc.RESET}", end="", flush=True)
        env = os.environ.copy()
        env["VITE_API_URL"] = f"http://localhost:{backend_port}"
        # strictPort = fail-fast if port is taken (so our "did it die?" check works)
        cmd = ["npm", "run", "dev", "--", "--port", str(port), "--strictPort"]
        proc = _spawn(cmd, cwd=APP_DIR / "frontend", env=env, logfile=logfile)
        self.frontend_process = proc  # Immediately visible to signal handler
        time.sleep(0.5)
        print(
            f"\r  {AnsiEsc.DIM}▸ Waiting for frontend to serve...{AnsiEsc.RESET}",
            end="",
            flush=True,
        )
        self.wait_to_serve(port)
        print(
            f"\r  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} Frontend started (pid {proc.pid}, port {port})        "
        )

    def cleanup(self) -> None:
        """Cleanup all running processes (process groups)."""
        if self.cleanup_called:
            return
        self.cleanup_called = True

        print("\n👋 Shutting down...", flush=True)

        # Kill all process groups immediately with SIGKILL for reliability
        for proc in (self.backend_process, self.frontend_process):
            if proc and proc.poll() is None:
                # Kill the process group
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

                # Also kill the direct process as fallback
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    proc.kill()

        # Brief wait for processes to die
        for proc in (self.backend_process, self.frontend_process):
            if proc:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=0.3)

    def monitor_child_liveness(self) -> None:
        while True:
            if self.backend_process and self.backend_process.poll() is not None:
                print(
                    f"\n{AnsiEsc.RED}✗{AnsiEsc.RESET} Backend process died unexpectedly",
                    file=sys.stderr,
                )
                print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
                sys.exit(1)
            if self.frontend_process and self.frontend_process.poll() is not None:
                print(
                    f"\n{AnsiEsc.RED}✗{AnsiEsc.RESET} Frontend process died unexpectedly",
                    file=sys.stderr,
                )
                print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1.0)

    def run(self) -> None:
        """Main entry point to run the development servers."""
        print(f"{AnsiEsc.DIM}Logfile: {LOGFILE}{AnsiEsc.RESET}")
        print(f"{AnsiEsc.DIM}Finding available ports...{AnsiEsc.RESET}")

        backend_port = find_available_port(BACKEND_DEFAULT_START)
        frontend_port = find_available_port(FRONTEND_DEFAULT_START)
        print(f" - {AnsiEsc.DIM}Backend port: {backend_port}{AnsiEsc.RESET}")
        print(f" - {AnsiEsc.DIM}Frontend port: {frontend_port}{AnsiEsc.RESET}")
        print()

        print(f"{AnsiEsc.BOLD}🚀 Starting development servers{AnsiEsc.RESET}")
        print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}")

        # Open logfile for streaming child output
        with open(LOGFILE, "a", buffering=1, encoding="utf-8") as logfile:
            self.start_backend(backend_port, logfile)
            self.start_frontend(frontend_port, backend_port, logfile)

            print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}\n")

            # Success banner
            print(f"{AnsiEsc.GREEN}{AnsiEsc.BOLD}✓ Ready!{AnsiEsc.RESET}\n")
            print(f"{AnsiEsc.DIM}Backend   http://localhost:{backend_port}/{AnsiEsc.RESET}")
            print(
                f"{AnsiEsc.BOLD}Frontend  {AnsiEsc.GREEN}{AnsiEsc.BOLD}{AnsiEsc.UNDERLINE}http://localhost:{frontend_port}/{AnsiEsc.RESET}\n"
            )
            print(f"{AnsiEsc.DIM}  Press Ctrl+C to stop{AnsiEsc.RESET}")

            # Monitor child liveness
            self.monitor_child_liveness()


def main() -> None:
    LOGFILE.unlink(missing_ok=True)
    with open(LOGFILE, "w", encoding="utf-8") as lf:
        lf.write(f"Launcher started at {datetime.now().isoformat()}\n")

    # Create runner and register signal handlers
    runner = AppRunner()

    def signal_handler(_signum: int, _frame: FrameType | None) -> None:
        """Handle termination signals by cleaning up and exiting."""
        runner.cleanup()
        sys.exit(0)

    # Register cleanup handlers
    atexit.register(runner.cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    # Run the app
    runner.run()


if __name__ == "__main__":
    main()
