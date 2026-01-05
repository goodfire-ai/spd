# SLURM Refactoring Uncertainties and Decisions

This document records the design decisions from the SLURM utilities refactoring.

## Decisions Made

### 1. Standardized preamble: `set -euo pipefail`

All generated SLURM scripts now include `set -euo pipefail` at the start, added by slurm.py. This ensures:
- `-e`: Exit immediately if a command fails
- `-u`: Treat unset variables as an error
- `-o pipefail`: Pipeline returns the exit status of the last failing command

Callers no longer need to add this themselves.

### 2. Custom echoes in commands

Callers can add their own custom echo statements in the command string for job-specific debugging info. For example, harvest adds:
```bash
echo "=== Harvest ==="
echo "WANDB_PATH: {wandb_path}"
...
```

This keeps slurm.py generic while allowing callers to customize logging.

### 3. Removed `source_env_file` option

Python scripts that need environment variables (like OPENROUTER_API_KEY for autointerp) should load them themselves using python-dotenv or similar, rather than having the SLURM script source .env files.

### 4. Removed `slurm_logs_dir` parameter

The `slurm_logs_dir` parameter was removed from `compute_utils.create_slurm_array_script`. All SLURM utilities now use `SLURM_LOGS_DIR` from settings directly. No backwards compatibility was needed since all users pull latest code.

### 5. Multi-node DDP env vars handling

Environment variables from `Command.env_vars` are prepended to the command string as `export K=V && {cmd}` rather than having slurm.py handle them specially. This keeps slurm.py generic.

### 6. CPU-only jobs (n_gpus=0)

Generate `--gres=gpu:0` for CPU-only jobs. SLURM handles this correctly.

## Things Not Changed

1. **Git snapshot logic**: Still handled separately in `git_utils.py`
2. **WandB setup**: Still in respective launcher scripts
3. **TrainingJob handling**: Still in compute_utils.py
4. **Command building**: Still specific to each launcher

## Testing Notes

- All 313 tests pass
- Type checking passes with no errors or warnings
- Verified generated scripts contain `set -euo pipefail`
