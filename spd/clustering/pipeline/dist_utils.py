"""Distribution utilities for parallel clustering via subprocess shell-out."""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import IO

from spd.log import logger
from spd.settings import REPO_ROOT


def launch_child_with_json_fd(cmd: list[str]) -> tuple[subprocess.Popen[bytes], IO[bytes]]:
    """Launch child process with JSON fd via environment variable.

    This allows the child to write structured JSON output to a dedicated file descriptor
    while still allowing stdout/stderr to stream normally to the console.

    Args:
        cmd: Command and arguments to execute

    Returns:
        Tuple of (subprocess handle, read file descriptor for JSON results)
    """
    json_r_fd, json_w_fd = os.pipe()
    os.set_inheritable(json_w_fd, True)
    os.set_inheritable(json_r_fd, False)

    # Pass the fd number via environment variable
    env: dict[str, str] = dict(os.environ)
    env["JSON_FD"] = str(json_w_fd)

    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,  # Let stdout stream to console
        stderr=None,  # Let stderr stream to console
        pass_fds=(json_w_fd,),
        close_fds=True,
    )

    # In parent process: close the write fd (child has it) and return read fd
    os.close(json_w_fd)
    json_r: IO[bytes] = os.fdopen(json_r_fd, "rb", buffering=0)
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
    - Tasks are round-robin assigned to devices
    - Each device can have up to workers_per_device tasks running concurrently

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
    if log_fn is None:
        log_fn = logger.info
    if log_fn_error is None:
        log_fn_error = lambda msg: logger.error(msg)

    n_devices: int = len(devices)
    if n_devices == 0:
        raise ValueError("devices must be non-empty")

    # Track active processes per device to enforce workers_per_device limit
    device_active_counts: dict[str, int] = {device: 0 for device in devices}
    active: list[tuple[subprocess.Popen[bytes], IO[bytes], Path, str]] = []
    results: list[dict[str, str | None]] = []

    n_files: int = len(data_files)
    try:
        for idx, dataset in enumerate(data_files):
            # Find next device with capacity using round-robin starting point
            device_idx = idx % n_devices
            attempts = 0
            while device_active_counts[devices[device_idx]] >= workers_per_device:
                # Wait for any process to finish if all devices are at capacity
                if all(count >= workers_per_device for count in device_active_counts.values()):
                    proc_to_wait, json_r_to_wait, dataset_path, device_to_free = active[0]
                    result = _read_json_result(json_r_to_wait, dataset_path)
                    proc_to_wait.wait()
                    results.append(result)
                    device_active_counts[device_to_free] -= 1
                    log_fn(f"Process {proc_to_wait.pid} finished, freeing slot on {device_to_free}")
                    active.pop(0)

                # Try next device
                device_idx = (device_idx + 1) % n_devices
                attempts += 1
                if attempts >= n_devices:
                    # We've checked all devices, start from beginning
                    attempts = 0

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
            active.append((proc, json_r, dataset, device))
            device_active_counts[device] += 1
            log_fn(
                f"Started clustering {idx + 1}/{n_files} on {device} (pid={proc.pid}, active on device: {device_active_counts[device]}/{workers_per_device})\n\t{dataset}"
            )

        # Wait for remaining processes
        for proc, json_r, dataset_path, device in active:
            result = _read_json_result(json_r, dataset_path)
            proc.wait()
            results.append(result)
            device_active_counts[device] -= 1
            log_fn(f"Process {proc.pid} finished on {device}")

    except Exception as e:
        log_fn_error(f"An error occurred: {e}")
        for proc, json_r, _, _ in active:
            proc.kill()
            json_r.close()
            log_fn_error(f"Killed process {proc.pid} due to error")
        raise e

    return results
