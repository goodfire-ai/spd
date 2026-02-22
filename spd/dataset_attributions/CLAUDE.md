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
# Single GPU (defaults from DatasetAttributionConfig, auto-generates subrun ID)
python -m spd.dataset_attributions.scripts.run_worker <wandb_path>

# Single GPU with config
python -m spd.dataset_attributions.scripts.run_worker <wandb_path> --config_json '{"n_batches": 500}'

# Multi-GPU (run in parallel via shell, tmux, etc.)
# All workers and the merge step must share the same --subrun_id
SUBRUN="da-$(date +%Y%m%d_%H%M%S)"
python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 1000}' --rank 0 --world_size 4 --subrun_id $SUBRUN &
python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 1000}' --rank 1 --world_size 4 --subrun_id $SUBRUN &
python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 1000}' --rank 2 --world_size 4 --subrun_id $SUBRUN &
python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 1000}' --rank 3 --world_size 4 --subrun_id $SUBRUN &
wait

# Merge results after all workers complete
python -m spd.dataset_attributions.scripts.run_merge <path> --subrun_id $SUBRUN
```

Each worker processes batches where `batch_idx % world_size == rank`, then the merge step combines all partial results.

## Data Storage

Each attribution invocation creates a timestamped sub-run directory. `AttributionRepo` automatically loads from the latest sub-run.

```
SPD_OUT_DIR/dataset_attributions/<run_id>/
├── da-20260211_120000/                    # sub-run 1
│   ├── dataset_attributions.pt            # Final merged attributions
│   └── worker_states/                     # cleaned up after merge
│       └── dataset_attributions_rank_*.pt
├── da-20260211_140000/                    # sub-run 2
│   └── ...
```

Legacy layout (pre sub-run) is still supported as a fallback by `AttributionRepo`:

```
SPD_OUT_DIR/dataset_attributions/<run_id>/
└── dataset_attributions.pt
```

## Architecture

### SLURM Launcher (`scripts/run_slurm.py`, `scripts/run_slurm_cli.py`)

Entry point via `spd-attributions`. Submits array job + dependent merge job.

### Worker Script (`scripts/run_worker.py`)

Harvests attributions for a single GPU. Called by SLURM array jobs or run directly.
- `--config_json`: Provide `DatasetAttributionConfig` as JSON (defaults used if omitted)
- `--rank R --world_size N`: Process subset of batches
- `--subrun_id`: Sub-run identifier (auto-generated if not provided)

### Merge Script (`scripts/run_merge.py`)

Combines per-rank attribution files into a single merged result.
- `--subrun_id`: Which sub-run to merge (required)

### Config (`config.py`)

`DatasetAttributionConfig` (tuning params) and `AttributionsSlurmConfig` (DatasetAttributionConfig + SLURM params). `wandb_path` is a runtime arg, not part of config.

### Harvest Logic (`harvest.py`)

Main harvesting functions:
- `harvest_attributions(wandb_path, config, output_dir, ...)`: Process batches for a single rank
- `merge_attributions(output_dir)`: Combine worker results from `output_dir/worker_states/` into `output_dir`

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

### Repository (`repo.py`)

`AttributionRepo` provides read access via `AttributionRepo.open(run_id)`. Returns `None` if no data exists. Storage is loaded eagerly at construction.

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
