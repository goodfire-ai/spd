## Plan for Adding MPI/DDP Support to spd/scripts/run.py

### Summary
Add distributed data parallel (DDP) support for LM experiments with minimal code changes. The implementation will add a `--dp` flag to specify the number of GPUs for data parallelism, modify command generation to use `mpirun` when needed, and update SLURM script generation to allocate multiple GPUs.

### Key Design Decisions
1. **Validation at the earliest point**: Check dp compatibility with experiments in the main function
2. **Pass dp parameter through the call chain**: Add dp parameter to functions that need it
3. **Modify command generation**: Wrap commands with mpirun only for lm experiments when dp > 1
4. **Update SLURM GPU allocation**: Change from `--gres=gpu:1` to `--gres=gpu:<dp>`
5. **Reject local mode with DDP**: Raise error if dp > 1 and local mode is enabled

### Implementation Steps

#### 1. Update main function in spd/scripts/run.py
- Add `dp: int = 1` parameter with validation (1-8 range)
- Add early validation: 
  - If dp > 1 and local mode, raise error
  - If dp > 1, check that all experiments are lm experiments
- Pass dp parameter to `generate_commands` and `create_slurm_array_script`

#### 2. Update generate_commands function
- Add `dp: int = 1` parameter
- For each experiment, check if it's an lm experiment (task_name == "lm")
- When dp > 1 and experiment is lm, wrap the python command with `mpirun -np {dp}`
- Example: `mpirun -np 4 python spd/experiments/lm/lm_decomposition.py ...`

#### 3. Update create_slurm_array_script in spd/utils/slurm_utils.py
- Add `dp: int = 1` parameter
- Change GPU allocation from hardcoded `--gres=gpu:1` to `--gres=gpu:{dp}`
- Only apply when not in CPU mode

### Code Changes

#### spd/scripts/run.py changes:

1. **main function signature** (line ~584):
```python
def main(
    experiments: str | None = None,
    ...
    dp: int = 1,  # NEW: Add dp parameter after 'cpu' parameter
    ...
) -> None:
```

2. **main function validation** (after line ~658):
```python
# Validate dp parameter
if dp < 1 or dp > 8:
    raise ValueError(f"dp must be between 1 and 8, got {dp}")

# Cannot use DDP with local mode
if dp > 1 and local:
    raise ValueError("DDP (dp > 1) is not supported in local mode")

# If dp > 1, ensure all experiments are lm experiments  
if dp > 1:
    non_lm_experiments = [
        exp for exp in experiments_list 
        if EXPERIMENT_REGISTRY[exp].task_name != "lm"
    ]
    if non_lm_experiments:
        raise ValueError(
            f"DDP (dp > 1) is only supported for lm experiments. "
            f"Non-lm experiments found: {non_lm_experiments}"
        )
```

3. **generate_commands call** (line ~710):
```python
commands: list[str] = generate_commands(
    experiments_list=experiments_list,
    run_id=run_id,
    sweep_params_file=sweep_params_file,
    project=project,
    dp=dp,  # NEW: Pass dp parameter
)
```

4. **generate_commands function signature** (line ~377):
```python
def generate_commands(
    experiments_list: list[str],
    run_id: str,
    sweep_params_file: str | None = None,
    project: str = "spd",
    dp: int = 1,  # NEW: Add dp parameter
) -> list[str]:
```

5. **generate_commands command building** (lines ~417-420 and ~446-451):
For the fixed configuration run (around line 417):
```python
# Check if this is an lm experiment and dp > 1
is_lm = config_entry.task_name == "lm"
if dp > 1 and is_lm:
    command = (
        f"mpirun -np {dp} python {decomp_script} '{config_json}' "
        f"--sweep_id {run_id} --evals_id {experiment}"
    )
else:
    command = (
        f"python {decomp_script} '{config_json}' "
        f"--sweep_id {run_id} --evals_id {experiment}"
    )
```

For the parameter sweep run (around line 446):
```python
# Check if this is an lm experiment and dp > 1
is_lm = config_entry.task_name == "lm"
if dp > 1 and is_lm:
    command = (
        f"mpirun -np {dp} python {decomp_script} '{config_json}' "
        f"--sweep_id {run_id} "
        f"--evals_id {experiment} "
        f"--sweep_params_json '{sweep_params_json}'"
    )
else:
    command = (
        f"python {decomp_script} '{config_json}' "
        f"--sweep_id {run_id} "
        f"--evals_id {experiment} "
        f"--sweep_params_json '{sweep_params_json}'"
    )
```

6. **create_slurm_array_script call** (line ~730):
```python
create_slurm_array_script(
    script_path=array_script,
    job_name=job_name,
    commands=commands,
    cpu=cpu,
    snapshot_branch=snapshot_branch,
    max_concurrent_tasks=n_agents,
    dp=dp,  # NEW: Pass dp parameter
)
```

#### spd/utils/slurm_utils.py changes:

1. **create_slurm_array_script function signature** (line ~25):
```python
def create_slurm_array_script(
    script_path: Path,
    job_name: str,
    commands: list[str],
    snapshot_branch: str,
    cpu: bool = False,
    time_limit: str = "72:00:00",
    max_concurrent_tasks: int | None = None,
    dp: int = 1,  # NEW: Add dp parameter
) -> None:
```

2. **GPU configuration** (line ~46):
```python
gpu_config = "#SBATCH --gres=gpu:0" if cpu else f"#SBATCH --gres=gpu:{dp}"
```

### Docstring Updates

Update the main function docstring to include:
```python
dp: Number of GPUs for data parallelism (1-8). Only supported for lm experiments.
    Cannot be used with local mode (default: 1)
```

Update the usage examples in the module docstring to include:
```python
spd-run --experiments ss_mlp --dp 4  # Run with 4 GPUs for data parallelism
```

### Testing Strategy
1. Test with dp=1 (default) - should work exactly as before
2. Test with dp=4 for an lm experiment (ss_mlp) - should generate mpirun command
3. Test with dp=2 for non-lm experiment - should raise error
4. Test with dp=0 or dp=9 - should raise validation error
5. Test with dp=2 and --local - should raise error
6. Verify SLURM script has correct GPU allocation (--gres=gpu:4 for dp=4)

### Benefits of This Approach
- **Minimal code changes**: Only touches necessary functions
- **Early validation**: Fails fast if invalid configuration
- **Backward compatible**: Default dp=1 preserves existing behavior
- **Clean separation**: DDP logic is isolated to command generation
- **Future extensibility**: Easy to add DDP support for other experiments later
- **Clear error messages**: Users get immediate feedback on invalid configurations

### Progress Tracking

- [x] Create PLAN.md file with implementation plan
- [x] Update main function in spd/scripts/run.py - add dp parameter and validation
- [x] Update generate_commands function to support mpirun wrapping
- [x] Update create_slurm_array_script in slurm_utils.py to allocate multiple GPUs
- [x] Run tests with make test - All tests pass
- [x] Run type checker with make type - No errors
- [x] Run formatter with make format - Code formatted

### Implementation Complete

The MPI/DDP support has been successfully added to spd/scripts/run.py. The implementation:

1. Added `dp` parameter (default=1) to specify number of GPUs for data parallelism
2. Validates that dp is between 1-8 and raises error if dp>1 with local mode
3. Ensures only lm experiments can use dp>1 (raises error otherwise)
4. Wraps lm experiment commands with `mpirun -np {dp}` when dp>1
5. Updates SLURM scripts to allocate the correct number of GPUs (`--gres=gpu:{dp}`)
6. Maintains backward compatibility (dp=1 by default preserves existing behavior)

All tests pass, type checking succeeds, and the code has been formatted according to project standards.