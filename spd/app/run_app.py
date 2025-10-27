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
from pathlib import Path
from types import FrameType
from typing import TextIO
from urllib.error import URLError

# ANSI color codes
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
DIM = "\033[2m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

# Configuration
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
        f"{RED}✗{RESET} Could not find available port after checking 100 ports from {start_port}",
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
                    f"\r  {RED}✗{RESET} {service_name} failed to start within {timeout}s        ",
                    file=sys.stderr,
                )
                print(f"{DIM}Check {LOGFILE} for details{RESET}", file=sys.stderr)
                sys.exit(1)
            time.sleep(0.5)


def start_backend(port: int, logfile: TextIO) -> subprocess.Popen[str]:
    """Start the backend server."""
    print(f"{DIM}  ▸ Starting backend...{RESET}", end="", flush=True)

    project_root = APP_DIR.parent.parent
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        str(project_root / "spd" / "app" / "backend" / "controller.py"),
        "--port",
        str(port),
    ]

    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid if sys.platform != "win32" else None,
    )

    # Wait for backend to be healthy
    wait_for_service(f"http://localhost:{port}", STARTUP_TIMEOUT_SECONDS, "Backend")
    print(f"\r  {GREEN}✓{RESET} Backend started (port {port})        ")

    return process


def install_frontend_deps() -> None:
    """Install frontend dependencies."""
    print(f"{DIM}  ▸ Installing frontend dependencies...{RESET}", end="", flush=True)

    frontend_dir = APP_DIR / "frontend"
    result = subprocess.run(["npm", "ci"], cwd=frontend_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, ["npm", "ci"], result.stderr)
    
    print(f"\r  {GREEN}✓{RESET} Frontend dependencies installed        ")


def start_frontend(port: int, backend_port: int, logfile: TextIO) -> subprocess.Popen[str]:
    """Start the frontend server."""
    print(f"{DIM}  ▸ Starting frontend...{RESET}", end="", flush=True)

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
        preexec_fn=os.setsid if sys.platform != "win32" else None,
    )

    # Wait for frontend to be healthy
    wait_for_service(f"http://localhost:{port}", STARTUP_TIMEOUT_SECONDS, "Frontend")
    print(f"\r  {GREEN}✓{RESET} Frontend started (port {port})        ")

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

    # Find available ports
    print(f"{DIM}Finding available ports...{RESET}")
    backend_port = find_available_port(start_port=3000)
    frontend_port = find_available_port(start_port=5173)

    print(f"{DIM}  Backend port: {backend_port}{RESET}")
    print(f"{DIM}  Frontend port: {frontend_port}{RESET}")
    print()

    # Fancy header
    print(f"{BOLD}🚀 Starting development servers{RESET}")
    print(f"{DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

    # Start services
    logfile.write(f"Backend started (port {backend_port})\n")
    logfile.write(f"Frontend started (port {frontend_port})\n")

    backend_process = start_backend(backend_port, logfile)
    logfile.write(f"Backend started (pid {backend_process.pid})\n")
    install_frontend_deps()
    frontend_process = start_frontend(frontend_port, backend_port, logfile)
    logfile.write(f"Frontend started (pid {frontend_process.pid})\n")

    print(f"{DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

    # Success message
    print()
    print(f"{GREEN}{BOLD}✓ Ready!{RESET}")
    print()
    print(f"{DIM}Backend   http://localhost:{backend_port}/{RESET}")
    print(f"{BOLD}Frontend  {GREEN}{BOLD}{UNDERLINE}http://localhost:{frontend_port}/{RESET}")
    print(f"{BOLD}Logfile   {UNDERLINE}{LOGFILE}{RESET}")
    print()
    print(f"{DIM}  Press Ctrl+C to stop{RESET}")

    # Wait indefinitely - cleanup will be handled by atexit/signal handlers
    try:
        while True:
            # Check if processes are still alive
            if backend_process.poll() is not None:
                print(f"\n{RED}✗{RESET} Backend process died unexpectedly", file=sys.stderr)
                print(f"{DIM}Check {LOGFILE} for details{RESET}", file=sys.stderr)
                sys.exit(1)
            if frontend_process.poll() is not None:
                print(f"\n{RED}✗{RESET} Frontend process died unexpectedly", file=sys.stderr)
                print(f"{DIM}Check {LOGFILE} for details{RESET}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
