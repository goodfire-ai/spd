# Harvest Module

Offline GPU pipeline that collects component statistics in a single pass over training data. Produces data consumed by the autointerp module (`spd/autointerp/`) and the app (`spd/app/`).

## Usage

```bash
# Local execution
python -m spd.harvest.scripts.run_harvest <wandb_path> \
    --n_batches 1000 \
    --batch_size 256 \
    --context_length 512

# Via SLURM (supports multi-GPU parallelism)
spd-harvest <wandb_path>
```

## Data Storage

Data is stored in `SPD_OUT_DIR/harvest/` (see `spd/settings.py`):

```
SPD_OUT_DIR/harvest/<run_id>/
├── activation_contexts/
│   ├── config.json
│   ├── summary.json          # Lightweight {component_key: {layer, idx, mean_ci}}
│   └── components.jsonl      # One ComponentData per line (~4GB, streamed on demand)
└── correlations/
    ├── component_correlations.pt
    └── token_stats.pt
```

## Architecture

### Harvest (`harvest.py`)

Main entry point. Orchestrates the `Harvester` to process batches and saves results.

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
- `load_activation_context_single(run_id, component_key)` -> ComponentData (streams file)
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