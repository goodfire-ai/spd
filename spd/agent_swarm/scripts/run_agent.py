"""Worker script that runs inside each SLURM job.

This script:
1. Creates an isolated output directory for this agent
2. Starts the app backend with an isolated database
3. Loads the SPD run
4. Launches Claude Code with investigation instructions
5. Handles cleanup on exit
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

import fire
import requests

from spd.agent_swarm.agent_prompt import get_agent_prompt
from spd.agent_swarm.schemas import SwarmEvent
from spd.agent_swarm.scripts.run_slurm import get_swarm_output_dir
from spd.log import logger


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts}"
    )


def wait_for_backend(port: int, timeout: float = 120.0) -> bool:
    """Wait for the backend to become healthy."""
    url = f"http://localhost:{port}/api/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


def load_run(port: int, wandb_path: str, context_length: int) -> bool:
    """Load the SPD run into the backend."""
    url = f"http://localhost:{port}/api/runs/load"
    params = {"wandb_path": wandb_path, "context_length": context_length}
    try:
        resp = requests.post(url, params=params, timeout=300)
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Failed to load run: {e}")
        return False


def log_event(events_path: Path, event: SwarmEvent) -> None:
    """Append an event to the events log."""
    with open(events_path, "a") as f:
        f.write(event.model_dump_json() + "\n")


def run_agent(
    wandb_path: str,
    task_id: int,
    swarm_id: str,
    context_length: int = 128,
    max_turns: int = 50,
) -> None:
    """Run a single investigation agent.

    Args:
        wandb_path: WandB path of the SPD run.
        task_id: SLURM task ID (1-indexed).
        swarm_id: Unique identifier for this swarm.
        context_length: Context length for prompts.
        max_turns: Maximum agentic turns before stopping (prevents runaway agents).
    """
    # Setup output directory
    swarm_dir = get_swarm_output_dir(swarm_id)
    task_dir = swarm_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    events_path = task_dir / "events.jsonl"
    explanations_path = task_dir / "explanations.jsonl"
    db_path = task_dir / "app.db"

    # Initialize empty output files
    explanations_path.touch()

    log_event(
        events_path,
        SwarmEvent(
            event_type="start",
            message=f"Agent {task_id} starting",
            details={"wandb_path": wandb_path, "swarm_id": swarm_id},
        ),
    )

    # Find available port (offset by task_id to reduce collisions)
    port = find_available_port(start_port=8000 + (task_id - 1) * 10)
    logger.info(f"[Task {task_id}] Using port {port}")

    log_event(
        events_path,
        SwarmEvent(
            event_type="progress",
            message=f"Starting backend on port {port}",
            details={"port": port, "db_path": str(db_path)},
        ),
    )

    # Start backend with isolated database
    env = os.environ.copy()
    env["SPD_APP_DB_PATH"] = str(db_path)

    backend_cmd = [
        sys.executable,
        "-m",
        "spd.app.backend.server",
        "--port",
        str(port),
    ]

    backend_proc = subprocess.Popen(
        backend_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Setup cleanup handler
    def cleanup(signum: int | None = None, frame: FrameType | None = None) -> None:
        _ = frame  # Unused but required by signal handler signature
        logger.info(f"[Task {task_id}] Cleaning up...")
        if backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        # Wait for backend to be ready
        logger.info(f"[Task {task_id}] Waiting for backend...")
        if not wait_for_backend(port):
            log_event(
                events_path,
                SwarmEvent(
                    event_type="error",
                    message="Backend failed to start",
                ),
            )
            raise RuntimeError("Backend failed to start")

        logger.info(f"[Task {task_id}] Backend ready, loading run...")
        log_event(
            events_path,
            SwarmEvent(
                event_type="progress",
                message="Backend ready, loading run",
            ),
        )

        # Load the SPD run
        if not load_run(port, wandb_path, context_length):
            log_event(
                events_path,
                SwarmEvent(
                    event_type="error",
                    message="Failed to load run",
                    details={"wandb_path": wandb_path},
                ),
            )
            raise RuntimeError(f"Failed to load run: {wandb_path}")

        logger.info(f"[Task {task_id}] Run loaded, launching Claude Code...")
        log_event(
            events_path,
            SwarmEvent(
                event_type="progress",
                message="Run loaded, launching Claude Code agent",
            ),
        )

        # Generate agent prompt
        agent_prompt = get_agent_prompt(
            port=port,
            wandb_path=wandb_path,
            task_id=task_id,
            output_dir=str(task_dir),
        )

        # Write prompt to file for reference
        prompt_path = task_dir / "agent_prompt.md"
        prompt_path.write_text(agent_prompt)

        # Launch Claude Code with streaming JSON output
        claude_output_path = task_dir / "claude_output.jsonl"
        claude_cmd = [
            "claude",
            "--print",
            "--verbose",  # Required for stream-json output
            "--output-format",
            "stream-json",
            "--max-turns",
            str(max_turns),
            "--dangerously-skip-permissions",
        ]

        logger.info(f"[Task {task_id}] Starting Claude Code (max_turns={max_turns})...")
        logger.info(f"[Task {task_id}] Monitor with: tail -f {claude_output_path}")
        logger.info(
            f"[Task {task_id}] Parse with: tail -f {claude_output_path} | jq -r '.result // empty'"
        )

        # Open output file for streaming writes
        with open(claude_output_path, "w") as output_file:
            claude_proc = subprocess.Popen(
                claude_cmd,
                stdin=subprocess.PIPE,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(task_dir),
            )

            # Send the investigation prompt and close stdin
            investigation_request = f"""
{agent_prompt}

---

Please begin your investigation:

1. **FIRST**: Create `{task_dir}/research_log.md` with a header like "# Research Log - Task {task_id}"
2. Check the backend status and explore component interpretations
3. Find an interesting behavior to investigate
4. **Update research_log.md frequently** with your progress, findings, and next steps

Remember:
- research_log.md is your PRIMARY output - humans will read this to follow your work
- Update it every few minutes with what you're doing and discovering
- Write complete explanations to explanations.jsonl when you finish investigating a behavior
"""
            assert claude_proc.stdin is not None
            claude_proc.stdin.write(investigation_request)
            claude_proc.stdin.close()

            # Wait for Claude to finish (output streams to file in real-time)
            claude_proc.wait()

        log_event(
            events_path,
            SwarmEvent(
                event_type="complete",
                message="Investigation complete",
                details={"exit_code": claude_proc.returncode},
            ),
        )

        logger.info(f"[Task {task_id}] Investigation complete")

    except Exception as e:
        log_event(
            events_path,
            SwarmEvent(
                event_type="error",
                message=f"Agent failed: {e}",
                details={"error_type": type(e).__name__},
            ),
        )
        logger.error(f"[Task {task_id}] Failed: {e}")
        raise
    finally:
        cleanup()


def cli() -> None:
    fire.Fire(run_agent)


if __name__ == "__main__":
    cli()
