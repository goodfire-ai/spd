# Dataset Attributions Module

Multi-GPU pipeline for computing component-to-component attribution strengths aggregated over the training dataset. Unlike prompt attributions (single-prompt, position-aware), dataset attributions answer: "In aggregate, which components typically influence each other?"

## Usage

```bash
# Submit 8-GPU SLURM job
spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8

# Submit 24-GPU SLURM job
spd-attributions <wandb_path> --n_batches 2000 --n_gpus 24

# With optional parameters
spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8 \
    --batch_size 64 --ci_threshold 1e-6 --time 48:00:00 --max_concurrent 12
```

The command:
1. Creates a git snapshot branch for reproducibility (jobs may be queued)
2. Submits a SLURM job array with N tasks (one per GPU)
3. Each task processes batches where `batch_idx % world_size == rank`
4. Submits a merge job (depends on array completion) that combines all worker results

## Data Storage

Data is stored in `SPD_OUT_DIR/dataset_attributions/` (see `spd/settings.py`):

```
SPD_OUT_DIR/dataset_attributions/<run_id>/
├── dataset_attributions.pt           # Final merged attributions
└── dataset_attributions_rank_*.pt    # Per-worker results (cleaned up after merge)
```

## Architecture

### SLURM Launcher (`scripts/run_slurm.py`, `scripts/run_slurm_cli.py`)

Entry point via `spd-attributions`. Submits array job + dependent merge job.

### Worker Script (`scripts/run.py`)

Internal script called by SLURM jobs. Supports:
- `--rank R --world_size N`: Process subset of batches
- `--merge`: Combine per-rank results into final file

### Harvest Logic (`harvest.py`)

Main harvesting functions:
- `harvest_attributions()`: Process batches for a single rank
- `merge_attributions()`: Combine results from all ranks

### Attribution Harvester (`harvester.py`)

Core class that accumulates attribution strengths using gradient × activation formula:

```
attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]
```

Key optimizations:
1. Sum outputs over positions before gradients (reduces backward passes)
2. For output targets, compute attributions to pre-unembed residual then multiply by W_unembed

### Storage (`storage.py`)

`DatasetAttributionStorage` class for loading/saving attribution matrices.

Matrix structure:
- Rows (sources): wte tokens `[0, vocab_size)` + component layers `[vocab_size, ...)`
- Cols (targets): component layers `[0, n_components)` + output tokens `[n_components, ...)`

### Loaders (`loaders.py`)

```python
from spd.dataset_attributions.loaders import load_dataset_attributions

storage = load_dataset_attributions(run_id)
if storage:
    # Get top sources attributing to a component
    top_sources = storage.get_top_sources("h.0.mlp.fc1:5", k=10, sign="positive")

    # Get top targets a component attributes to
    top_targets = storage.get_top_targets("h.0.mlp.fc1:5", k=10, sign="positive")
```

## Key Types

```python
DatasetAttributionStorage   # Main storage class with attribution matrix
DatasetAttributionEntry     # Single entry: component_key, layer, component_idx, value
DatasetAttributionConfig    # Config: wandb_path, n_batches, batch_size, ci_threshold
```
