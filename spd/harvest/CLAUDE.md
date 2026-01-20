# Harvest Module

Offline GPU pipeline that collects component statistics in a single pass over training data. Produces data consumed by the autointerp module (`spd/autointerp/`) and the app (`spd/app/`).

## Usage (SLURM)

```bash
# Submit 24-GPU SLURM job
spd-harvest <wandb_path> --n_batches 2000 --n_gpus 24

# With optional parameters
spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8 \
    --batch_size 256 --ci_threshold 1e-6 --time 24:00:00 --max_concurrent 12 --job_suffix 30m
```

The command:
1. Creates a git snapshot branch for reproducibility (jobs may be queued)
2. Submits a SLURM job array with N tasks (one per GPU)
3. Each task processes batches where `batch_idx % world_size == rank`
4. Submits a merge job (depends on array completion) that combines all worker results

## Usage (non-SLURM)

For environments without SLURM, run the worker script directly:

```bash
# Single GPU
python -m spd.harvest.scripts.run <wandb_path> --n_batches 1000

# Multi-GPU (run in parallel via shell, tmux, etc.)
python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 0 --world_size 4 &
python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 1 --world_size 4 &
python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 2 --world_size 4 &
python -m spd.harvest.scripts.run <path> --n_batches 1000 --rank 3 --world_size 4 &
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

Internal script called by SLURM jobs. Supports:
- `--rank R --world_size N`: Process subset of batches
- `--merge`: Combine per-rank results into final files

### Harvest Logic (`harvest.py`)

Main harvesting functions:
- `harvest_activation_contexts()`: Process batches for a single rank
- `merge_activation_contexts()`: Combine results from all ranks

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