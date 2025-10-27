"""
Development server launcher for SPD app.
Starts both backend and frontend servers with automatic port detection and graceful cleanup.
"""

import atexit
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
from urllib.error import URLError


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


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use on any interface (IPv4 or IPv6)."""
    # Check IPv4
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True

    # Check IPv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("::1", port))
            except OSError:
                return True
    except OSError:
        # IPv6 not available on this system
        pass

    return False


def find_available_port(start_port: int) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        if not is_port_in_use(port):
            return port

    print(
        f"{AnsiEsc.RED}✗{AnsiEsc.RESET} Could not find available port after checking 100 ports from {start_port}",
        file=sys.stderr,
    )
    sys.exit(1)


def wait_for_service(url: str, timeout: int, service_name: str) -> None:
    """Wait for a service to become available."""
    import urllib.request

    start_time = time.time()
    while True:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except (URLError, OSError):
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print(
                    f"\r  {AnsiEsc.RED}✗{AnsiEsc.RESET} {service_name} failed to start within {timeout}s        ",
                    file=sys.stderr,
                )
                print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
                sys.exit(1)
            time.sleep(0.5)


def start_backend(port: int, logfile: TextIO) -> subprocess.Popen[str]:
    """Start the backend server."""
    print(f"{AnsiEsc.DIM}  ▸ Starting backend...{AnsiEsc.RESET}", end="", flush=True)

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        str(APP_DIR / "backend" / "server.py"),
        "--port",
        str(port),
    ]

    project_root = APP_DIR.parent.parent
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,  # isolate child process group so cleanup signals do not hit launcher
    )

    # Wait for backend to be healthy
    wait_for_service(f"http://localhost:{port}", STARTUP_TIMEOUT_SECONDS, "Backend")
    print(f"\r  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} Backend started (port {port})        ")

    return process


def start_frontend(port: int, backend_port: int, logfile: TextIO) -> subprocess.Popen[str]:
    """Start the frontend server."""
    print(f"{AnsiEsc.DIM}  ▸ Starting frontend...{AnsiEsc.RESET}", end="", flush=True)

    frontend_dir = APP_DIR / "frontend"
    env = os.environ.copy()
    env["VITE_API_URL"] = f"http://localhost:{backend_port}"

    cmd = ["npm", "run", "dev", "--", "--port", str(port)]

    process = subprocess.Popen(
        cmd,
        cwd=frontend_dir,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,  # isolate child process group so cleanup signals do not hit launcher
    )

    # Wait for frontend to be healthy
    wait_for_service(f"http://localhost:{port}", STARTUP_TIMEOUT_SECONDS, "Frontend")
    print(f"\r  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} Frontend started (port {port})        ")

    return process


def exit(_signum: int, _frame: FrameType | None) -> None:
    """Handle interrupt signals."""
    sys.exit(0)  # Will trigger atexit cleanup


def main():
    """Main entry point."""
    # Initialize logfile
    LOGFILE.unlink(missing_ok=True)
    LOGFILE.touch()

    backend_process: subprocess.Popen[str] | None = None
    frontend_process: subprocess.Popen[str] | None = None
    logfile = open(LOGFILE, "w", buffering=1)  # noqa: SIM115

    def cleanup():
        """Cleanup all running processes."""
        print("\n👋 Shutting down...")

        # Terminate process groups gracefully
        if backend_process and backend_process.poll() is None:
            os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
        if frontend_process and frontend_process.poll() is None:
            os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)

        # Give processes time to cleanup gracefully
        time.sleep(1.0)

        # Force kill process groups if still running
        if backend_process and backend_process.poll() is None:
            os.killpg(os.getpgid(backend_process.pid), signal.SIGKILL)
        if frontend_process and frontend_process.poll() is None:
            os.killpg(os.getpgid(frontend_process.pid), signal.SIGKILL)

        logfile.close()

    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)
    print(f"{AnsiEsc.DIM}logfile: {LOGFILE}{AnsiEsc.RESET}")

    # Find available ports
    print(f"{AnsiEsc.DIM}Finding available ports...{AnsiEsc.RESET}")
    backend_port = find_available_port(start_port=8000)
    frontend_port = find_available_port(start_port=5173)

    print(f"{AnsiEsc.DIM}  Backend port: {backend_port}{AnsiEsc.RESET}")
    print(f"{AnsiEsc.DIM}  Frontend port: {frontend_port}{AnsiEsc.RESET}")
    print()

    # Fancy header
    print(f"{AnsiEsc.BOLD}🚀 Starting development servers{AnsiEsc.RESET}")
    print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}")

    # Start services
    logfile.write(f"Backend started (port {backend_port})\n")
    logfile.write(f"Frontend started (port {frontend_port})\n")

    backend_process = start_backend(backend_port, logfile)
    logfile.write(f"Backend started (pid {backend_process.pid})\n")

    frontend_process = start_frontend(frontend_port, backend_port, logfile)
    logfile.write(f"Frontend started (pid {frontend_process.pid})\n")

    print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}")

    # Success message
    print()
    print(f"{AnsiEsc.GREEN}{AnsiEsc.BOLD}✓ Ready!{AnsiEsc.RESET}")
    print()
    print(f"{AnsiEsc.DIM}Backend   http://localhost:{backend_port}/{AnsiEsc.RESET}")
    print(
        f"{AnsiEsc.BOLD}Frontend  {AnsiEsc.GREEN}{AnsiEsc.BOLD}{AnsiEsc.UNDERLINE}http://localhost:{frontend_port}/{AnsiEsc.RESET}"
    )
    print()
    print(f"{AnsiEsc.DIM}  Press Ctrl+C to stop{AnsiEsc.RESET}")

    # Wait indefinitely - cleanup will be handled by atexit/signal handlers
    try:
        while True:
            # Check if processes are still alive
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
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
