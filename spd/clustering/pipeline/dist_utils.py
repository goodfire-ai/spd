"""Distribution utilities for parallel clustering via subprocess shell-out."""

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from spd.log import logger
from spd.settings import REPO_ROOT


@dataclass
class ActiveProcess:
    """Tracks an active subprocess and its associated metadata."""

    proc: subprocess.Popen[bytes]
    json_fd: IO[bytes]
    dataset_path: Path
    device: str


def launch_child_with_json_fd(cmd: list[str]) -> tuple[subprocess.Popen[bytes], IO[bytes]]:
    """Launch child process with JSON fd via environment variable.

    This allows the child to write structured JSON output to a dedicated file descriptor
    while still allowing stdout/stderr to stream normally to the console.

    Args:
        cmd: Command and arguments to execute

    Returns:
        Tuple of (subprocess handle, read file descriptor for JSON results)
    """
    # get the pipes
    json_fd_rw: tuple[int, int] = os.pipe()  # (read_fd, write_fd)
    os.set_inheritable(json_fd_rw[1], True)
    os.set_inheritable(json_fd_rw[0], False)

    # Pass the fd number via environment variable
    env: dict[str, str] = dict(os.environ)
    env["JSON_FD"] = str(json_fd_rw[1])

    # launch the child process
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,  # Let stdout stream to console
        stderr=None,  # Let stderr stream to console
        pass_fds=(json_fd_rw[1],),
        close_fds=True,
    )

    # In parent process: close the write fd (child has it) and return read fd
    os.close(json_fd_rw[1])
    json_r: IO[bytes] = os.fdopen(json_fd_rw[0], "rb", buffering=0)
    return proc, json_r


def _open_json_fd() -> IO[str]:
    """Open file descriptor for JSON output from environment variable.

    Called by child processes to get the fd for emitting structured results.
    """
    fd_num: int = int(os.environ["JSON_FD"])
    return os.fdopen(fd_num, "w", buffering=1)


def emit_result(obj: dict[str, str | None]) -> None:
    """Emit result JSON via environment fd.

    Called by child processes to return structured results to the parent.

    Args:
        obj: Result dictionary to serialize and emit
    """
    out: IO[str] = _open_json_fd()
    print(json.dumps(obj, separators=(",", ":")), file=out, flush=True)


def _read_json_result(json_r: IO[bytes], dataset_path: Path) -> dict[str, str | None]:
    """Read JSON result from file descriptor.

    Args:
        json_r: Read file descriptor for JSON data
        dataset_path: Path to dataset being processed (for error messages)

    Returns:
        Parsed JSON result dictionary

    Raises:
        RuntimeError: If no JSON result was received
        ValueError: If JSON parsing failed
    """
    json_line: bytes = json_r.readline()
    if not json_line:
        raise RuntimeError(f"No JSON result received from {dataset_path.stem}")

    json_str: str = json_line.decode().strip()
    try:
        result: dict[str, str | None] = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON result from {dataset_path.stem}: {e}\nJSON string: {json_str}"
        ) from e


def distribute_clustering(
    config_path: Path,
    data_files: list[Path],
    devices: list[str],
    base_path: Path,
    run_identifier: str,
    workers_per_device: int = 1,
    log_fn: Callable[[str], None] | None = None,
    log_fn_error: Callable[[str], None] | None = None,
) -> list[dict[str, str | None]]:
    """Distribute clustering tasks across multiple devices via subprocess.

    Launches clustering processes using shell-out approach with JSON fd for structured
    results. Manages concurrency based on workers_per_device and available devices.

    The concurrency model:
    - Total concurrency = workers_per_device x len(devices)
    - Uses round-robin device assignment starting point
    - If target device is full, uses any available device
    - If all devices are full, waits for a process on the target device to finish

    Args:
        config_path: Path to clustering configuration file
        data_files: List of batch data files to process
        devices: List of device strings (e.g., ['cuda:0', 'cuda:1'])
        base_path: Base directory for clustering outputs
        run_identifier: Unique identifier for this clustering run
        workers_per_device: Maximum concurrent workers per device
        log_fn: Optional logging function for info messages
        log_fn_error: Optional logging function for error messages

    Returns:
        List of result dictionaries from each batch processing

    Raises:
        ValueError: If devices list is empty
        RuntimeError: If subprocess fails or doesn't return results
    """
    # setup logger
    if log_fn is None:
        log_fn = logger.info
    if log_fn_error is None:
        log_fn_error = lambda msg: logger.error(msg)

    # check devices
    n_devices: int = len(devices)
    if n_devices == 0:
        raise ValueError("devices must be non-empty")

    # Track active processes per device to enforce workers_per_device limit
    device_active_counts: dict[str, int] = {device: 0 for device in devices}
    active: list[ActiveProcess] = []
    results: list[dict[str, str | None]] = []

    n_files: int = len(data_files)
    try:
        for idx, dataset in enumerate(data_files):
            # Find a device with capacity, starting from round-robin position
            device_idx = idx % n_devices

            # Check if we need to wait for a device to free up
            while all(count >= workers_per_device for count in device_active_counts.values()):
                # All devices are at capacity - wait for ANY process to finish
                log_fn(
                    f"All devices at capacity ({workers_per_device} workers each). Waiting for any process to finish..."
                )

                # Wait for the first process (any device)
                active_proc = active[0]
                result = _read_json_result(active_proc.json_fd, active_proc.dataset_path)
                active_proc.proc.wait()
                results.append(result)
                device_active_counts[active_proc.device] -= 1
                log_fn(
                    f"Process {active_proc.proc.pid} finished, freeing slot on {active_proc.device}"
                )
                active.pop(0)

            # Now find a device with capacity, starting from our round-robin position
            for i in range(n_devices):
                check_idx = (device_idx + i) % n_devices
                if device_active_counts[devices[check_idx]] < workers_per_device:
                    device_idx = check_idx
                    break

            device: str = devices[device_idx]

            cmd: list[str] = [
                "uv",
                "run",
                "python",
                str(REPO_ROOT / "spd/clustering/pipeline/s2_clustering.py"),
                "--config",
                str(config_path),
                "--dataset-path",
                str(dataset),
                "--base-path",
                str(base_path),
                "--run-identifier",
                run_identifier,
                "--device",
                device,
            ]
            log_fn("[cmd] " + " ".join(cmd))

            proc, json_r = launch_child_with_json_fd(cmd)
            active_proc = ActiveProcess(
                proc=proc, json_fd=json_r, dataset_path=dataset, device=device
            )
            active.append(active_proc)
            device_active_counts[device] += 1
            log_fn(
                f"Started clustering {idx + 1}/{n_files} on {device} (pid={proc.pid}, active on device: {device_active_counts[device]}/{workers_per_device})\n\t{dataset}"
            )

        # Wait for remaining processes
        for active_proc in active:
            result = _read_json_result(active_proc.json_fd, active_proc.dataset_path)
            active_proc.proc.wait()
            results.append(result)
            device_active_counts[active_proc.device] -= 1
            log_fn(f"Process {active_proc.proc.pid} finished on {active_proc.device}")

    except Exception as e:
        log_fn_error(f"An error occurred: {e}")
        for active_proc in active:
            active_proc.proc.kill()
            active_proc.json_fd.close()
            log_fn_error(f"Killed process {active_proc.proc.pid} due to error")
        raise e

    return results
