# Dataset Attributions Module

Multi-GPU pipeline for computing component-to-component attribution strengths aggregated over the training dataset. Unlike prompt attributions (single-prompt, position-aware), dataset attributions answer: "In aggregate, which components typically influence each other?"

## Usage (SLURM)

```bash
# Process specific number of batches
spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8

# Process entire training dataset (omit --n_batches)
spd-attributions <wandb_path> --n_gpus 24

# With optional parameters
spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8 \
    --batch_size 64 --ci_threshold 1e-6 --time 48:00:00
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
# Single GPU (defaults from DatasetAttributionConfig)
python -m spd.dataset_attributions.scripts.run <wandb_path>

# Single GPU with config file
python -m spd.dataset_attributions.scripts.run <wandb_path> --config_path path/to/config.yaml

# Multi-GPU (run in parallel via shell, tmux, etc.)
python -m spd.dataset_attributions.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 0 --world_size 4 &
python -m spd.dataset_attributions.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 1 --world_size 4 &
python -m spd.dataset_attributions.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 2 --world_size 4 &
python -m spd.dataset_attributions.scripts.run <path> --config_json '{"n_batches": 1000}' --rank 3 --world_size 4 &
wait

# Merge results after all workers complete
python -m spd.dataset_attributions.scripts.run <path> --merge
```

Each worker processes batches where `batch_idx % world_size == rank`, then the merge step combines all partial results.

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

Internal script called by SLURM jobs. Accepts config via `--config_path` (file) or `--config_json` (inline JSON). Supports:
- `--config_path`/`--config_json`: Provide `DatasetAttributionConfig` (defaults used if neither given)
- `--rank R --world_size N`: Process subset of batches
- `--merge`: Combine per-rank results into final file

### Config (`config.py`)

`DatasetAttributionConfig` (tuning params) and `AttributionsSlurmConfig` (DatasetAttributionConfig + SLURM params). `wandb_path` is a runtime arg, not part of config.

### Harvest Logic (`harvest.py`)

Main harvesting functions:
- `harvest_attributions(wandb_path, config, ...)`: Process batches for a single rank
- `merge_attributions(wandb_path)`: Combine results from all ranks

### Attribution Harvester (`harvester.py`)

Core class that accumulates attribution strengths using gradient × activation formula:

```
attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]
```

Key optimizations:
1. Sum outputs over positions before gradients (reduces backward passes)
2. For output targets, store attributions to output residual stream instead of vocab tokens (reduces storage from O((V+C)²) to O((V+C)×(C+d_model)))

### Storage (`storage.py`)

`DatasetAttributionStorage` class using output-residual-based storage for scalability.

**Storage structure:**
- `source_to_component`: (n_sources, n_components) - direct attributions to component targets
- `source_to_out_residual`: (n_sources, d_model) - attributions to output residual stream for output queries

**Source indexing (rows):**
- `[0, vocab_size)`: wte tokens
- `[vocab_size, vocab_size + n_components)`: component layers

**Target handling:**
- Component targets: direct lookup in `source_to_component`
- Output targets: computed on-the-fly via `source_to_out_residual @ w_unembed[:, token_id]`

**Why output-residual-based storage?**

For large vocab models (V=32K), the naive approach would require O((V+C)²) storage (~4 GB).
The output-residual-based approach requires only O((V+C)×(C+d)) storage (~670 MB for Llama-scale),
a 6.5x reduction. Output attributions are computed on-the-fly at query time with negligible latency.

### Loaders (`loaders.py`)

```python
from spd.dataset_attributions.loaders import load_dataset_attributions

storage = load_dataset_attributions(run_id)
if storage:
    # Get top sources attributing to a component (no w_unembed needed)
    top_sources = storage.get_top_sources("h.0.mlp.c_fc:5", k=10, sign="positive")

    # Get top component targets (no w_unembed needed)
    top_comp_targets = storage.get_top_component_targets("h.0.mlp.c_fc:5", k=10, sign="positive")

    # Get top targets including outputs (requires w_unembed)
    w_unembed = model.target_model.lm_head.weight.T.detach()
    top_targets = storage.get_top_targets("h.0.mlp.c_fc:5", k=10, sign="positive", w_unembed=w_unembed)

    # Get top output targets only (requires w_unembed)
    top_outputs = storage.get_top_output_targets("h.0.mlp.c_fc:5", k=10, sign="positive", w_unembed=w_unembed)
```

## Key Types

```python
DatasetAttributionStorage   # Main storage class with split matrices
DatasetAttributionEntry     # Single entry: component_key, layer, component_idx, value
DatasetAttributionConfig    # Config (BaseConfig): n_batches, batch_size, ci_threshold
```

## Query Methods

| Method | w_unembed required? | Description |
|--------|---------------------|-------------|
| `get_top_sources(component_key, k, sign)` | No | Top sources → component target |
| `get_top_sources(output_key, k, sign, w_unembed)` | Yes | Top sources → output token |
| `get_top_component_targets(source_key, k, sign)` | No | Top component targets |
| `get_top_output_targets(source_key, k, sign, w_unembed)` | Yes | Top output token targets |
| `get_top_targets(source_key, k, sign, w_unembed)` | Yes | All targets (components + outputs) |
| `get_attribution(source_key, component_key)` | No | Single component attribution |
| `get_attribution(source_key, output_key, w_unembed)` | Yes | Single output attribution |
