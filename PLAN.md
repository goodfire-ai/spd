# SLURM Utilities Refactoring Plan

## Main Goal

**Eliminate duplication in SLURM job submission code by creating a single unified module `spd/utils/slurm.py`.**

The codebase has 4+ different places that generate and submit SLURM scripts, each with ~80% identical code. This refactoring consolidates all SLURM utilities into one module while keeping SPD-specific training logic (multi-node DDP, `TrainingJob`) separate in `compute_utils.py`.

**Guiding principles:**
- Keep `slurm.py` simple and generic (no SPD-specific types)
- SPD training complexity stays in `compute_utils.py`
- Prefer returning strings over writing files (let callers decide)
- When in doubt, choose the simpler option

---

## Current State

### Files That Generate SLURM Scripts

| File | Purpose | How It Generates Scripts |
|------|---------|--------------------------|
| `spd/utils/compute_utils.py` | SPD training jobs | `create_slurm_array_script()` - takes `TrainingJob` objects, handles multi-node DDP |
| `spd/utils/command_utils.py` | Clustering pipeline | `create_slurm_array_script()` and `create_slurm_script()` - takes command strings |
| `spd/harvest/scripts/run_slurm.py` | Harvest SLURM launcher | Inline f-string (~40 lines) |
| `spd/autointerp/scripts/run_slurm.py` | Autointerp SLURM launcher | Inline f-string (~35 lines) |

### Shared Code Pattern (Duplicated Everywhere)

All scripts share this structure:
```bash
#!/bin/bash
#SBATCH --job-name=...
#SBATCH --partition=...
#SBATCH --nodes=...
#SBATCH --gres=gpu:...
#SBATCH --time=...
#SBATCH --output=.../slurm-%j.out

# Workspace setup
WORK_DIR="$HOME/slurm_workspaces/..."
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

# Git setup (either clone+checkout or just cd)
git clone <repo> "$WORK_DIR"
cd "$WORK_DIR"
git checkout "<snapshot_branch>"

# Venv
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --no-dev --link-mode copy -q
source .venv/bin/activate

# Run command
<command>
```

### Post-Submit Pattern (Duplicated Everywhere)

```python
script_path.write_text(content)
script_path.chmod(0o755)
job_id = submit_slurm_script(script_path)
final_path = SBATCH_SCRIPTS_DIR / f"prefix_{job_id}.sh"
script_path.rename(final_path)
(SLURM_LOGS_DIR / f"slurm-{job_id}.out").touch()
```

---

## New Module: `spd/utils/slurm.py`

### Public API

```python
"""Unified SLURM job submission utilities.

This module provides a single source of truth for generating and submitting SLURM jobs.
It handles:
- SBATCH header generation
- Workspace creation with cleanup
- Git snapshot checkout (optional)
- Virtual environment activation
- Job submission with script renaming and log file creation

For SPD-specific training jobs with multi-node DDP, see compute_utils.py which
uses this module internally.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SlurmConfig:
    """Configuration for a SLURM job.

    Attributes:
        job_name: Name for the SLURM job (appears in squeue)
        partition: SLURM partition to submit to
        n_gpus: Number of GPUs per node (0 for CPU-only jobs)
        n_nodes: Number of nodes (default 1)
        time: Time limit in HH:MM:SS format
        snapshot_branch: Git branch to checkout. If None, just cd to REPO_ROOT without cloning.
        dependency_job_id: If set, job waits for this job to complete (afterok dependency)
        source_env_file: If True, source .env file before running command (for API keys)
    """
    job_name: str
    partition: str
    n_gpus: int = 1
    n_nodes: int = 1
    time: str = "72:00:00"
    snapshot_branch: str | None = None
    dependency_job_id: str | None = None
    source_env_file: bool = False


@dataclass
class SlurmArrayConfig(SlurmConfig):
    """Configuration for a SLURM job array.

    Attributes:
        max_concurrent_tasks: Maximum number of array tasks to run concurrently.
                              If None, no limit (all tasks can run at once).
    """
    max_concurrent_tasks: int | None = None


@dataclass
class SubmitResult:
    """Result of submitting a SLURM job.

    Attributes:
        job_id: The SLURM job ID (string, e.g., "12345")
        script_path: Path where the script was saved (renamed to include job ID)
        log_pattern: Human-readable log path pattern for display
    """
    job_id: str
    script_path: Path
    log_pattern: str


def generate_script(config: SlurmConfig, command: str) -> str:
    """Generate a single SLURM job script.

    Args:
        config: SLURM job configuration
        command: The shell command to run

    Returns:
        Complete SLURM script content as a string
    """
    ...


def generate_array_script(config: SlurmArrayConfig, commands: list[str]) -> str:
    """Generate a SLURM job array script.

    Each command in the list becomes one array task. Commands are executed via
    a case statement based on SLURM_ARRAY_TASK_ID.

    Args:
        config: SLURM array job configuration
        commands: List of shell commands, one per array task

    Returns:
        Complete SLURM array script content as a string
    """
    ...


def submit(
    script_content: str,
    script_name_prefix: str,
    is_array: bool = False,
    n_array_tasks: int | None = None,
) -> SubmitResult:
    """Write script to disk, submit to SLURM, and set up logging.

    This function:
    1. Writes script to SBATCH_SCRIPTS_DIR with a temporary name
    2. Submits via sbatch
    3. Renames script to include the SLURM job ID
    4. Creates empty log file(s) for tailing

    Args:
        script_content: The SLURM script content
        script_name_prefix: Prefix for script filename (e.g., "harvest", "clustering")
        is_array: Whether this is an array job (affects log file creation)
        n_array_tasks: Number of array tasks (required if is_array=True)

    Returns:
        SubmitResult with job ID, script path, and log pattern
    """
    ...


def run_locally(
    commands: list[str],
    parallel: bool = False,
    track_resources: bool = False,
) -> dict[str, dict[str, float]] | None:
    """Run commands locally instead of via SLURM.

    Useful for testing and for --local mode in clustering pipeline.

    Args:
        commands: List of shell commands to run
        parallel: If True, run all commands in parallel. If False, run sequentially.
        track_resources: If True, track and return resource usage via /usr/bin/time

    Returns:
        If track_resources is True, dict mapping commands to resource metrics.
        Otherwise None.
    """
    ...
```

### Internal Implementation Details

The module should have these internal helpers:

```python
def _generate_sbatch_header(
    config: SlurmConfig,
    is_array: bool = False,
    array_range: str | None = None,
) -> str:
    """Generate the #SBATCH directive block.

    Handles:
    - --job-name, --partition, --nodes, --gres, --time, --output
    - --array (for array jobs)
    - --dependency (if dependency_job_id is set)
    """
    ...


def _generate_setup_section(config: SlurmConfig) -> str:
    """Generate workspace creation and git/venv setup.

    If snapshot_branch is set:
    - Create workspace dir with trap for cleanup
    - Clone repo to workspace
    - Copy .env file
    - Checkout snapshot branch
    - uv sync and activate venv

    If snapshot_branch is None:
    - Just cd to REPO_ROOT
    - Activate existing venv

    If source_env_file is True:
    - Source .env file (for API keys like OPENROUTER_API_KEY)
    """
    ...


def _generate_case_block(commands: list[str]) -> str:
    """Generate bash case statement for array jobs.

    SLURM arrays are 1-indexed, so command[0] goes in case 1).
    """
    ...


def _submit_script(script_path: Path) -> str:
    """Submit script via sbatch and return job ID.

    Raises RuntimeError if sbatch fails.
    """
    # This is the existing submit_slurm_script logic from command_utils.py
    ...
```

---

## Implementation Steps

### Step 1: Create `spd/utils/slurm.py`

Create the new module with the full API described above. Key implementation notes:

1. **SBATCH header generation**: Use f-strings, handle optional fields gracefully
   ```python
   lines = [
       f"#SBATCH --job-name={config.job_name}",
       f"#SBATCH --partition={config.partition}",
       f"#SBATCH --nodes={config.n_nodes}",
       f"#SBATCH --gres=gpu:{config.n_gpus}",
       f"#SBATCH --time={config.time}",
       f"#SBATCH --output={SLURM_LOGS_DIR}/slurm-%{'A_%a' if is_array else 'j'}.out",
   ]
   if is_array and array_range:
       lines.append(f"#SBATCH --array={array_range}")
   if config.dependency_job_id:
       lines.append(f"#SBATCH --dependency=afterok:{config.dependency_job_id}")
   ```

2. **Setup section**: Two modes based on `snapshot_branch`
   - With snapshot: Clone repo, checkout branch, uv sync
   - Without snapshot: Just cd to REPO_ROOT, source venv

3. **Log file output pattern**: Use `%A_%a` for arrays (job_array-task), `%j` for single jobs

4. **Copy `run_locally` from `command_utils.py`**: The `run_script_array_local` function with resource tracking

5. **Import from settings**: `REPO_ROOT`, `SLURM_LOGS_DIR`, `SBATCH_SCRIPTS_DIR`

### Step 2: Migrate Harvest Launcher

**File**: `spd/harvest/scripts/run_slurm.py`

**Before** (lines 70-94): Inline f-string building entire script

**After**:
```python
from spd.utils.slurm import SlurmConfig, generate_script, submit

def harvest(...):
    job_id = _generate_job_id()

    # Build command (keep existing command building logic)
    cmd_parts = [...]
    harvest_cmd = " \\\n    ".join(cmd_parts)

    # Generate and submit
    config = SlurmConfig(
        job_name=f"harvest-{job_id}",
        partition=partition,
        n_gpus=n_gpus or 1,
        time=time,
        snapshot_branch=None,  # Harvest doesn't use git snapshots
    )
    script_content = generate_script(config, harvest_cmd)
    result = submit(script_content, "harvest")

    # Logging (keep existing logger calls, update to use result.job_id, result.log_pattern)
    logger.section("Harvest job submitted!")
    logger.values({
        "Job ID": result.job_id,
        ...
    })
```

**Delete**: ~25 lines of inline script generation

### Step 3: Migrate Autointerp Launcher

**File**: `spd/autointerp/scripts/run_slurm.py`

**Before** (lines 52-83): Inline f-string with .env sourcing

**After**:
```python
from spd.utils.slurm import SlurmConfig, generate_script, submit

def launch_interpret_job(...):
    job_id = _generate_job_id()

    # Build command
    cmd_parts = [...]
    interpret_cmd = " \\\n    ".join(cmd_parts)

    # Generate and submit
    config = SlurmConfig(
        job_name=f"interpret-{job_id}",
        partition=partition,
        n_gpus=0,  # CPU-only
        time=time,
        snapshot_branch=None,
        source_env_file=True,  # For OPENROUTER_API_KEY
    )
    script_content = generate_script(config, interpret_cmd)
    result = submit(script_content, "interpret")

    # Logging...
```

**Delete**: ~30 lines of inline script generation

### Step 4: Migrate Clustering Pipeline

**File**: `spd/clustering/scripts/run_pipeline.py`

**Current imports**:
```python
from spd.utils.command_utils import (
    create_slurm_array_script,
    create_slurm_script,
    run_script_array_local,
    submit_slurm_script,
)
```

**New imports**:
```python
from spd.utils.slurm import (
    SlurmConfig,
    SlurmArrayConfig,
    generate_script,
    generate_array_script,
    submit,
    run_locally,
)
```

**Changes in `main()` function**:

1. **Local execution** (lines 285-315): Replace `run_script_array_local` with `run_locally`

2. **SLURM array job for clustering** (lines 326-337):
   ```python
   # Before
   create_slurm_array_script(
       script_path=clustering_script_path,
       job_name=...,
       commands=clustering_commands,
       snapshot_branch=execution_stamp.snapshot_branch,
       max_concurrent_tasks=pipeline_config.n_runs,
       n_gpus_per_job=1,
       partition=pipeline_config.slurm_partition,
   )
   array_job_id = submit_slurm_script(clustering_script_path)

   # After
   config = SlurmArrayConfig(
       job_name=f"{pipeline_config.slurm_job_name_prefix}_cluster",
       partition=pipeline_config.slurm_partition,
       n_gpus=1,
       snapshot_branch=execution_stamp.snapshot_branch,
       max_concurrent_tasks=pipeline_config.n_runs,
   )
   script_content = generate_array_script(config, clustering_commands)
   result = submit(script_content, "clustering", is_array=True, n_array_tasks=len(clustering_commands))
   array_job_id = result.job_id
   ```

3. **SLURM single jobs for calc_distances** (lines 346-361):
   ```python
   # Before
   create_slurm_script(
       script_path=calc_distances_script_path,
       job_name=...,
       command=cmd,
       snapshot_branch=execution_stamp.snapshot_branch,
       n_gpus=1,
       partition=pipeline_config.slurm_partition,
       dependency_job_id=array_job_id,
   )
   job_id = submit_slurm_script(calc_distances_script_path)

   # After
   config = SlurmConfig(
       job_name=f"{pipeline_config.slurm_job_name_prefix}_dist_{method}",
       partition=pipeline_config.slurm_partition,
       n_gpus=1,
       snapshot_branch=execution_stamp.snapshot_branch,
       dependency_job_id=array_job_id,
   )
   script_content = generate_script(config, cmd)
   result = submit(script_content, f"calc_distances_{method}")
   ```

4. **Remove tempfile usage**: The `submit()` function handles file writing, so no need for tempfile

### Step 5: Update `compute_utils.py` to Use `slurm.py`

**File**: `spd/utils/compute_utils.py`

Keep `TrainingJob`, `get_command()`, and the SPD-specific logic. Refactor `create_slurm_array_script` to use `slurm.py` internally.

**Before**:
- Manually builds entire script with SBATCH headers, setup section, case block
- Returns script content string

**After**:
```python
from spd.utils.slurm import SlurmArrayConfig, generate_array_script

def create_slurm_array_script(
    slurm_job_name: str,
    run_id: str,
    training_jobs: list[TrainingJob],
    sweep_params: dict[str, Any] | None,
    slurm_logs_dir: Path,  # Note: this becomes unused, slurm.py uses SLURM_LOGS_DIR
    snapshot_branch: str,
    n_gpus: int | None,
    partition: str,
    max_concurrent_tasks: int | None = None,
) -> str:
    """Create a SLURM job array script for SPD training jobs.

    This is a thin wrapper around slurm.generate_array_script that handles
    TrainingJob -> command string conversion and multi-node DDP setup.
    """
    # Convert TrainingJobs to command strings
    commands: list[str] = []
    for i, job in enumerate(training_jobs):
        cmd = get_command(run_id, job, i, n_gpus, sweep_params)
        # For multi-node, get_command returns Command with env_vars
        # We need to prepend env var exports to the command
        if cmd.env_vars:
            env_exports = " ".join(f"{k}={v}" for k, v in cmd.env_vars.items())
            full_cmd = f"export {env_exports} && {cmd.command}"
        else:
            full_cmd = cmd.command
        commands.append(full_cmd)

    # Calculate n_nodes for multi-node DDP
    if n_gpus is None or n_gpus <= GPUS_PER_NODE:
        n_nodes = 1
        gpus_per_node = n_gpus or 1
    else:
        n_nodes = n_gpus // GPUS_PER_NODE
        gpus_per_node = GPUS_PER_NODE

    config = SlurmArrayConfig(
        job_name=slurm_job_name,
        partition=partition,
        n_gpus=gpus_per_node,
        n_nodes=n_nodes,
        snapshot_branch=snapshot_branch,
        max_concurrent_tasks=max_concurrent_tasks,
    )

    return generate_array_script(config, commands)
```

**Note**: The multi-node DDP case with `srun` in `get_command()` already wraps the torchrun command with `srun bash -c`. This should continue to work since we're just passing the command string to `slurm.py`.

**Potential issue**: The current `compute_utils.create_slurm_array_script` sets `--ntasks={n_nodes}` for multi-node. Need to ensure `slurm.py` handles this. Add `n_tasks: int | None = None` to `SlurmConfig` if needed, defaulting to `n_nodes`.

### Step 6: Clean Up `command_utils.py`

**File**: `spd/utils/command_utils.py`

**Delete**:
- `create_slurm_array_script()` - moved to `slurm.py`
- `create_slurm_script()` - moved to `slurm.py`
- `submit_slurm_script()` - moved to `slurm.py`
- `run_script_array_local()` - moved to `slurm.py` as `run_locally()`

**After cleanup**: File should be empty or nearly empty. If empty, delete the file and remove any imports of it.

### Step 7: Update Imports Everywhere

Search for and update all imports:

```python
# Old
from spd.utils.command_utils import submit_slurm_script
from spd.utils.command_utils import create_slurm_array_script, create_slurm_script

# New
from spd.utils.slurm import submit, generate_script, generate_array_script
```

Files to check:
- `spd/scripts/run.py`
- `spd/harvest/scripts/run_slurm.py`
- `spd/autointerp/scripts/run_slurm.py`
- `spd/clustering/scripts/run_pipeline.py`

---

## Testing

### Manual Testing

1. **Harvest**: `spd-harvest <wandb_path> --n_batches 100` - verify job submits correctly
2. **Autointerp**: `spd-autointerp <wandb_path>` - verify CPU job with .env sourcing
3. **Clustering local**: `spd-clustering --config <path> --local` - verify local execution
4. **Clustering SLURM**: `spd-clustering --config <path>` - verify array job + dependent jobs
5. **SPD training**: `spd-run --experiments tms_5-2` - verify array job with git snapshot

### Automated Tests

Check if there are existing tests in `tests/` that cover SLURM utilities. If so, update them. If not, consider adding basic unit tests for:
- `generate_script()` output structure
- `generate_array_script()` case statement generation
- `_generate_sbatch_header()` with various configs

---

## Files Changed Summary

| File | Action |
|------|--------|
| `spd/utils/slurm.py` | **CREATE** - New unified module |
| `spd/utils/command_utils.py` | **DELETE** or gut (remove SLURM functions) |
| `spd/utils/compute_utils.py` | **MODIFY** - Use `slurm.py` internally |
| `spd/harvest/scripts/run_slurm.py` | **MODIFY** - Use `slurm.py` |
| `spd/autointerp/scripts/run_slurm.py` | **MODIFY** - Use `slurm.py` |
| `spd/clustering/scripts/run_pipeline.py` | **MODIFY** - Use `slurm.py` |
| `spd/scripts/run.py` | **MODIFY** - Update imports |

---

## Edge Cases to Handle

1. **CPU-only jobs**: `n_gpus=0` should produce `--gres=gpu:0` or omit the line entirely (check what SLURM expects)

2. **No snapshot branch**: When `snapshot_branch=None`, don't clone repo, just `cd $REPO_ROOT && source .venv/bin/activate`

3. **Multi-node ntasks**: For multi-node DDP, need `--ntasks={n_nodes}`. Either:
   - Add `n_tasks` to `SlurmConfig`
   - Or set `n_tasks = n_nodes` automatically

4. **Array job log file creation**: For array jobs, create multiple log files (`slurm-{job_id}_{i}.out` for i in 1..n_tasks)

5. **Empty commands list**: `generate_array_script([])` should raise an error

6. **Script permissions**: Set 0o755 on script files before submission
