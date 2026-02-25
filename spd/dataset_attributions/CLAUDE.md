# Dataset Attributions Module

Multi-GPU pipeline for computing component-to-component attribution strengths aggregated over the training dataset. Unlike prompt attributions (single-prompt, position-aware), dataset attributions answer: "In aggregate, which components typically influence each other?"

## Usage (SLURM)

```bash
spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8
spd-attributions <wandb_path> --n_gpus 24  # whole dataset
```

The command:
1. Creates a git snapshot branch for reproducibility
2. Submits a SLURM job array (one per GPU)
3. Each task processes batches where `batch_idx % world_size == rank`
4. Submits a merge job (depends on array completion)

## Usage (non-SLURM)

```bash
# Single GPU
python -m spd.dataset_attributions.scripts.run_worker <wandb_path>

# Multi-GPU
SUBRUN="da-$(date +%Y%m%d_%H%M%S)"
python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 1000}' --rank 0 --world_size 4 --subrun_id $SUBRUN &
python -m spd.dataset_attributions.scripts.run_worker <path> --config_json '{"n_batches": 1000}' --rank 1 --world_size 4 --subrun_id $SUBRUN &
# ...
wait
python -m spd.dataset_attributions.scripts.run_merge --wandb_path <path> --subrun_id $SUBRUN
```

## Data Storage

```
SPD_OUT_DIR/dataset_attributions/<run_id>/
├── da-20260223_183250/                    # sub-run (latest picked by repo)
│   ├── dataset_attributions.pt            # merged result
│   └── worker_states/
│       └── dataset_attributions_rank_*.pt
```

`AttributionRepo.open(run_id)` loads the latest `da-*` subrun that has a `dataset_attributions.pt`.

## Three Attribution Metrics

The harvester accumulates three metrics simultaneously:

| Metric | Formula | Description |
|--------|---------|-------------|
| `attr` | E[∂y/∂x · x] | Signed mean attribution |
| `attr_abs` | E[∂\|y\|/∂x · x] | Attribution to absolute value of target (2 backward passes) |
| `mean_squared_attr` | E[(∂y/∂x · x)²] | Mean squared attribution (pre-sqrt, mergeable across workers) |

Naming convention: modifier *before* `attr` applies to the target (e.g. `attr_abs` = attribution to |target|). Modifier *after* applies to the attribution itself (e.g. `squared_attr` = squared attribution).

## Architecture

### Storage (`storage.py`)

`DatasetAttributionStorage` stores three nested dicts:
```
attrs[target_layer][source_layer] = Tensor[target_d, source_d]
```

All layer names use **canonical addressing** (`"embed"`, `"0.glu.up"`, `"output"`).

For output targets, `target_d = d_model`. Output token attributions computed on-the-fly: `attr @ w_unembed[:, token_id]`.

Key methods: `get_attribution()`, `get_top_sources()`, `get_top_targets()` — all take an `AttrMetric` parameter to select which metric dict to query. `merge(paths)` classmethod for combining worker results.

### Harvester (`harvester.py`)

Accumulates attributions using gradient × activation. Uses **concrete module paths** internally (talks to model cache/CI). Key optimizations:
1. Sum outputs over positions before gradients (reduces backward passes)
2. Output-residual storage (O(d_model) instead of O(vocab))
3. `scatter_add_` for embed sources, vectorized `.add_()` for components (>14x faster than per-element loops)

### Harvest (`harvest.py`)

Orchestrates the pipeline: loads model, builds gradient connectivity, runs batches, translates concrete→canonical at storage boundary via `topology.target_to_canon()`.

### Scripts

- `scripts/run_worker.py` — worker entrypoint (single GPU)
- `scripts/run_merge.py` — merge entrypoint (CPU only, needs ~200G RAM)
- `scripts/run_slurm.py` — SLURM launcher (array + merge jobs)
- `scripts/run_slurm_cli.py` — CLI wrapper for `spd-attributions`

### Config (`config.py`)

- `DatasetAttributionConfig`: n_batches, batch_size, ci_threshold
- `AttributionsSlurmConfig`: adds n_gpus, partition, time, merge_time, merge_mem (default 200G)

### Repository (`repo.py`)

`AttributionRepo.open(run_id)` → loads latest subrun. Returns `None` if no data.

## Query Methods

All query methods take `metric: AttrMetric` (`"attr"`, `"attr_abs"`, or `"mean_squared_attr"`).

| Method | w_unembed? | Description |
|--------|-----------|-------------|
| `get_top_sources(target_key, k, sign, metric)` | If output target | Top sources → target |
| `get_top_targets(source_key, k, sign, metric)` | If include_outputs | Top targets ← source |
| `get_attribution(source_key, target_key, metric)` | If output target | Single attribution value |

Key format: `"embed:{token_id}"`, `"0.glu.up:{c_idx}"`, `"output:{token_id}"`.
