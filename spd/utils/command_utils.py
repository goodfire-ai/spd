"""Minimal utilities for running shell-safe commands locally and on SLURM."""

import subprocess
import tempfile
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT


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
            wrapped_cmd = (
                f'/usr/bin/time -f "K:%K M:%M P:%P S:%S U:%U e:%e" -o {resource_file} {cmd}'
            )
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


# =============================================================================
# SLURM utilities for simple command-based jobs (e.g., clustering pipeline)
# =============================================================================


def create_slurm_array_script(
    script_path: Path,
    job_name: str,
    commands: list[str],
    snapshot_branch: str | None,
    max_concurrent_tasks: int,
    n_gpus_per_job: int,
    partition: str,
) -> None:
    """Create a SLURM job array script from a list of commands.

    Args:
        script_path: Path to write the script to.
        job_name: Name for the SLURM job.
        commands: List of shell-safe command strings.
        snapshot_branch: Git branch to checkout (None to skip git checkout).
        max_concurrent_tasks: Maximum concurrent array tasks.
        n_gpus_per_job: Number of GPUs per job.
        partition: SLURM partition to use.
    """
    n_jobs = len(commands)
    array_range = f"1-{n_jobs}%{max_concurrent_tasks}"

    # Build case statement (SLURM arrays are 1-indexed)
    case_lines = []
    for i, cmd in enumerate(commands):
        case_lines.append(f"    {i + 1})")
        case_lines.append(f"        {cmd}")
        case_lines.append("        ;;")
    case_block = "\n".join(case_lines)

    # Git checkout section
    if snapshot_branch:
        git_section = f"""
# Clone the repository to the job-specific directory
git clone {REPO_ROOT} "$WORK_DIR"

# Change to the cloned repository directory
cd "$WORK_DIR"

# Copy the .env file from the original repository for WandB authentication
cp {REPO_ROOT}/.env .env

# Checkout the snapshot branch to ensure consistent code
git checkout "{snapshot_branch}"

# Ensure that dependencies are using the snapshot branch
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --no-dev --link-mode copy -q
source .venv/bin/activate
"""
    else:
        git_section = f"""
cd {REPO_ROOT}
source .venv/bin/activate
"""

    script_content = f"""\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{n_gpus_per_job}
#SBATCH --partition={partition}
#SBATCH --time=72:00:00
#SBATCH --job-name={job_name}
#SBATCH --output=$HOME/slurm_logs/slurm-%A_%a.out
#SBATCH --array={array_range}

# Create job-specific working directory
WORK_DIR="$HOME/slurm_workspaces/{job_name}-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$WORK_DIR"

# Clean up the workspace when the script exits
trap 'rm -rf "$WORK_DIR"' EXIT
{git_section}
echo "Running task $SLURM_ARRAY_TASK_ID..."

# Execute the appropriate command based on array task ID
case $SLURM_ARRAY_TASK_ID in
{case_block}
esac
"""

    script_path.write_text(script_content)


def create_slurm_script(
    script_path: Path,
    job_name: str,
    command: str,
    snapshot_branch: str | None,
    n_gpus: int,
    partition: str,
    dependency_job_id: str | None = None,
) -> None:
    """Create a single SLURM job script.

    Args:
        script_path: Path to write the script to.
        job_name: Name for the SLURM job.
        command: Shell-safe command string to run.
        snapshot_branch: Git branch to checkout (None to skip git checkout).
        n_gpus: Number of GPUs.
        partition: SLURM partition to use.
        dependency_job_id: If provided, this job will wait for the specified job to complete.
    """
    dependency_line = ""
    if dependency_job_id:
        dependency_line = f"#SBATCH --dependency=afterok:{dependency_job_id}"

    # Git checkout section
    if snapshot_branch:
        git_section = f"""
# Clone the repository to the job-specific directory
git clone {REPO_ROOT} "$WORK_DIR"

# Change to the cloned repository directory
cd "$WORK_DIR"

# Copy the .env file from the original repository for WandB authentication
cp {REPO_ROOT}/.env .env

# Checkout the snapshot branch to ensure consistent code
git checkout "{snapshot_branch}"

# Ensure that dependencies are using the snapshot branch
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --no-dev --link-mode copy -q
source .venv/bin/activate
"""
    else:
        git_section = f"""
cd {REPO_ROOT}
source .venv/bin/activate
"""

    script_content = f"""\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --partition={partition}
#SBATCH --time=72:00:00
#SBATCH --job-name={job_name}
#SBATCH --output=$HOME/slurm_logs/slurm-%j.out
{dependency_line}

# Create job-specific working directory
WORK_DIR="$HOME/slurm_workspaces/{job_name}-$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

# Clean up the workspace when the script exits
trap 'rm -rf "$WORK_DIR"' EXIT
{git_section}
echo "Running job..."
{command}
"""

    script_path.write_text(script_content)


def submit_slurm_script(script_path: Path) -> str:
    """Submit a SLURM script and return the job ID.

    Args:
        script_path: Path to the SLURM batch script.

    Returns:
        Job ID from submitted job.
    """
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit SLURM job: {result.stderr}")
    # Extract job ID from sbatch output (format: "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id
