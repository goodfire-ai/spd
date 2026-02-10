# Harvest Module

Offline GPU pipeline that collects component statistics in a single pass over training data. Produces data consumed by the autointerp module (`spd/autointerp/`) and the app (`spd/app/`).

## Usage (SLURM)

```bash
# Process specific number of batches
spd-harvest <wandb_path> --n_batches 2000 --n_gpus 24

# Process entire training dataset (omit --n_batches)
spd-harvest <wandb_path> --n_gpus 24

# With optional parameters
spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8 \
    --batch_size 256 --ci_threshold 1e-6 --time 24:00:00 --job_suffix 30m
```

The command:
1. Creates a git snapshot branch for reproducibility (jobs may be queued)
2. Submits a SLURM job array with N tasks (one per GPU)
3. Each task processes batches where `batch_idx % world_size == rank`
4. Submits a merge job (depends on array completion) that combines all worker results

**Note**: `--n_batches` is optional. If omitted, the pipeline processes the entire training dataset.

## Usage (non-SLURM)

For environments without SLURM, run the worker script directly:

```bash
# Single GPU (defaults from HarvestConfig)
python -m spd.harvest.scripts.run <wandb_path>

# Single GPU with config file
python -m spd.harvest.scripts.run <wandb_path> --config_path path/to/config.yaml

# Multi-GPU (run in parallel via shell, tmux, etc.)
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 0 --world_size 4 &
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 1 --world_size 4 &
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 2 --world_size 4 &
python -m spd.harvest.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 3 --world_size 4 &
wait

# Merge results after all workers complete
python -m spd.harvest.scripts.run <path> --merge
```

Each worker processes batches where `batch_idx % world_size == rank`, then the merge step combines all partial results.

## Data Storage

Data is stored in `SPD_OUT_DIR/harvest/` (see `spd/settings.py`):

```
SPD_OUT_DIR/harvest/<run_id>/
├── activation_contexts/
│   ├── config.json
│   ├── summary.json          # Lightweight {component_key: {layer, idx, mean_ci}}
│   └── components.jsonl      # One ComponentData per line (quite big - usually ≈10gb)
├── correlations/
│   ├── component_correlations.pt
│   └── token_stats.pt
└── worker_states/
    └── worker_*.pt           # Per-worker states (cleaned up after merge)
```

## Architecture

### SLURM Launcher (`scripts/run_slurm.py`, `scripts/run_slurm_cli.py`)

Entry point via `spd-harvest`. Submits array job + dependent merge job.

### Worker Script (`scripts/run.py`)

Internal script called by SLURM jobs. Accepts config via `--config_path` (file) or `--config_json` (inline JSON). Supports:
- `--config_path`/`--config_json`: Provide `HarvestConfig` (defaults used if neither given)
- `--rank R --world_size N`: Process subset of batches
- `--merge`: Combine per-rank results into final files

### Harvest Logic (`harvest.py`)

`HarvestConfig` (Pydantic `BaseConfig`) owns defaults for tuning params (batch_size, ci_threshold, etc.). `wandb_path` is a runtime arg passed separately.

Main harvesting functions:
- `harvest_activation_contexts(wandb_path, config, ...)`: Process batches for a single rank
- `merge_activation_contexts(wandb_path)`: Combine results from all ranks

### Harvester (`lib/harvester.py`)

Core class that accumulates statistics in a single pass:
- **Correlations**: Co-occurrence counts between components (for precision/recall/PMI)
- **Token stats**: Input token associations (hard counts) and output token associations (probability mass)
- **Activation examples**: Reservoir sampling for uniform coverage across dataset

Key optimizations:
- Reservoir sampling: O(1) per add, O(k) memory, uniform random sampling from stream
- Subsampling: Caps firings per batch at 10k (plenty for k=20 examples per component)
- All accumulation on GPU, only moves to CPU for final `build_results()`

### Reservoir Sampler (`lib/reservoir_sampler.py`)

Implements reservoir sampling for uniform random sampling from a stream. Maintains a fixed-size buffer of examples that represents a uniform sample over all items seen.

### Storage (`storage.py`)

`CorrelationStorage` and `TokenStatsStorage` classes for loading/saving harvested data.

### Loaders (`loaders.py`)

Functions for loading harvested data by run ID:
- `load_activation_contexts_summary(run_id)` -> dict[component_key, ComponentSummary]
- `load_component_activation_contexts(run_id, component_key)` -> ComponentData
- `load_correlations(run_id)` -> CorrelationStorage
- `load_token_stats(run_id)` -> TokenStatsStorage

## Key Types (`schemas.py`)

```python
ActivationExample     # Token window + CI values around a firing
ComponentData         # All harvested info for one component
ComponentTokenPMI     # Top/bottom tokens by PMI
```

## Analysis (`analysis.py`)

Query functions for exploring harvested data:
- Component correlations (precision, recall, Jaccard, PMI)
- Token statistics lookup
- Activation example retrieval