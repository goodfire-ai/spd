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
from shutil import which
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


def wait_to_serve(port: int) -> None:
    start = time.time()
    while time.time() < (start + 20.0):
        try:
            response = requests.get(f"http://localhost:{port}", timeout=1.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    print(f"{AnsiEsc.RED}✗{AnsiEsc.RESET} Healthcheck failed", file=sys.stderr)
    sys.exit(1)


def start_backend(port: int, logfile: TextIO, proc_ref: list[subprocess.Popen[str] | None]) -> subprocess.Popen[str]:
    print(f"  {AnsiEsc.DIM}▸ Starting backend...{AnsiEsc.RESET}", end="", flush=True)
    project_root = APP_DIR.parent.parent
    cmd = ["uv", "run", "python", "-u", str(APP_DIR / "backend" / "server.py"), "--port", str(port)]
    proc = _spawn(cmd, cwd=project_root, env=None, logfile=logfile)
    proc_ref[0] = proc  # Make visible to signal handler immediately
    time.sleep(0.5)
    print(f"\r  {AnsiEsc.DIM}▸ Waiting for backend to serve...{AnsiEsc.RESET}", end="", flush=True)
    wait_to_serve(port)
    print(
        f"\r  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} Backend started (pid {proc.pid}, port {port})        "
    )
    return proc


def start_frontend(port: int, backend_port: int, logfile: TextIO, proc_ref: list[subprocess.Popen[str] | None]) -> subprocess.Popen[str]:
    print(f"  {AnsiEsc.DIM}▸ Starting frontend...{AnsiEsc.RESET}", end="", flush=True)
    env = os.environ.copy()
    env["VITE_API_URL"] = f"http://localhost:{backend_port}"
    # strictPort = fail-fast if port is taken (so our "did it die?" check works)
    cmd = ["npm", "run", "dev", "--", "--port", str(port), "--strictPort"]
    proc = _spawn(cmd, cwd=APP_DIR / "frontend", env=env, logfile=logfile)
    proc_ref[0] = proc  # Make visible to signal handler immediately
    time.sleep(0.5)
    print(f"\r  {AnsiEsc.DIM}▸ Waiting for frontend to serve...{AnsiEsc.RESET}", end="", flush=True)
    wait_to_serve(port)
    print(
        f"\r  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} Frontend started (pid {proc.pid}, port {port})        "
    )
    return proc


def main() -> None:
    if which("npm") is None:
        print(
            f"{AnsiEsc.RED}✗ Missing dependency:{AnsiEsc.RESET} npm\n"
            f"{AnsiEsc.DIM}Install Node.js (includes npm): https://nodejs.org/{AnsiEsc.RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize logfile
    LOGFILE.unlink(missing_ok=True)
    with open(LOGFILE, "w", encoding="utf-8") as lf:
        lf.write(f"Launcher started at {datetime.now().isoformat()}\n")

    # Track processes - use lists so they can be mutated in nested functions before assignment
    backend_process_ref: list[subprocess.Popen[str] | None] = [None]
    frontend_process_ref: list[subprocess.Popen[str] | None] = [None]
    cleanup_called = False

    def cleanup() -> None:
        """Cleanup all running processes (process groups)."""
        nonlocal cleanup_called
        if cleanup_called:
            print("[DEBUG] Cleanup already called, skipping", flush=True)
            return
        cleanup_called = True

        backend_process = backend_process_ref[0]
        frontend_process = frontend_process_ref[0]

        print("\n👋 Shutting down...", flush=True)
        print(f"[DEBUG] Backend process: {backend_process.pid if backend_process else 'None'}", flush=True)
        print(f"[DEBUG] Frontend process: {frontend_process.pid if frontend_process else 'None'}", flush=True)

        # Kill all process groups immediately with SIGKILL for reliability
        # We skip SIGTERM since we need to be fast to avoid being killed ourselves
        for name, proc in [("backend", backend_process), ("frontend", frontend_process)]:
            if proc and proc.poll() is None:
                try:
                    pgid = os.getpgid(proc.pid)
                    print(f"[DEBUG] Killing {name} process group {pgid}", flush=True)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError) as e:
                    print(f"[DEBUG] Failed to kill {name} group: {e}", flush=True)

                # Also kill the direct process as fallback
                try:
                    proc.kill()
                    print(f"[DEBUG] Killed {name} process directly", flush=True)
                except (ProcessLookupError, PermissionError, OSError) as e:
                    print(f"[DEBUG] Failed to kill {name} directly: {e}", flush=True)

        # Brief wait for processes to die
        for name, proc in [("backend", backend_process), ("frontend", frontend_process)]:
            if proc:
                try:
                    proc.wait(timeout=0.3)
                    print(f"[DEBUG] {name} exited cleanly", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"[DEBUG] {name} did not exit in time", flush=True)

        print("[DEBUG] Cleanup complete", flush=True)

    def signal_handler(signum: int, _frame: FrameType | None) -> None:
        """Handle termination signals by cleaning up and exiting."""
        print(f"\n[DEBUG] Signal {signum} received!", flush=True)
        cleanup()
        print("[DEBUG] About to exit", flush=True)
        sys.exit(0)

    # Register cleanup handlers
    atexit.register(cleanup)
    print("[DEBUG] Registering signal handlers...", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal_handler)
        print(f"[DEBUG] Registered handler for signal {sig}", flush=True)

    print(f"{AnsiEsc.DIM}Logfile: {LOGFILE}{AnsiEsc.RESET}")
    print(f"{AnsiEsc.DIM}Finding available ports...{AnsiEsc.RESET}")
    print()

    backend_port = find_available_port(BACKEND_DEFAULT_START)
    frontend_port = find_available_port(FRONTEND_DEFAULT_START)

    print(f"{AnsiEsc.BOLD}🚀 Starting development servers{AnsiEsc.RESET}")
    print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}")

    # Open logfile for streaming child output
    with open(LOGFILE, "a", buffering=1, encoding="utf-8") as logfile:
        backend_process = start_backend(backend_port, logfile, backend_process_ref)
        frontend_process = start_frontend(frontend_port, backend_port, logfile, frontend_process_ref)

        print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}\n")

        # Success banner
        print(f"{AnsiEsc.GREEN}{AnsiEsc.BOLD}✓ Ready!{AnsiEsc.RESET}\n")
        print(f"{AnsiEsc.DIM}Backend   http://localhost:{backend_port}/{AnsiEsc.RESET}")
        print(
            f"{AnsiEsc.BOLD}Frontend  {AnsiEsc.GREEN}{AnsiEsc.BOLD}{AnsiEsc.UNDERLINE}http://localhost:{frontend_port}/{AnsiEsc.RESET}\n"
        )
        print(f"{AnsiEsc.DIM}  Press Ctrl+C to stop{AnsiEsc.RESET}")

        # Monitor child liveness
        while True:
            if backend_process.poll() is not None:
                print(
                    f"\n{AnsiEsc.RED}✗{AnsiEsc.RESET} Backend process died unexpectedly",
                    file=sys.stderr,
                )
                print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
                sys.exit(1)
            if frontend_process.poll() is not None:
                print(
                    f"\n{AnsiEsc.RED}✗{AnsiEsc.RESET} Frontend process died unexpectedly",
                    file=sys.stderr,
                )
                print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
