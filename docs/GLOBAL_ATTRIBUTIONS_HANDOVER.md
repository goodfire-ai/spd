# Global Attributions Feature - Handover Document

This document describes the global attributions feature: what it does, how it works, current status, and remaining work.

## Overview

**Goal**: Pre-compute attributions between components across an entire dataset during the harvest pipeline, then display them in the app.

**Why**: Local attributions (existing feature) show component interactions for a single prompt. Global attributions reveal **structural, dataset-wide patterns** - which components typically influence each other in aggregate.

**Status**: Feature is fully implemented but has performance issues that make it impractical for large models.

## Conceptual Background

### Local vs Global Attributions

| Aspect | Local Attributions | Global Attributions |
|--------|-------------------|---------------------|
| Computed | On-the-fly in app | Offline during harvest |
| Scope | Single prompt | Entire dataset |
| Question answered | "For this input, which components influenced this output?" | "In aggregate, which components typically influence each other?" |
| Formula | `grad(target) * activation(source)` for one input | Sum of `grad * activation` over all inputs |

### Attribution Calculation

For each (source, target) component pair, the attribution is:

```
attribution[source, target] = sum over dataset of (d_target / d_source) * activation_source
```

This uses **unmasked components + weight deltas** for gradient computation (per commit `b76e4d9e`), not the CI-masked computation used for display.

## Architecture

### Files Modified/Added

| File | Purpose |
|------|---------|
| `spd/harvest/lib/attribution_compute.py` | Core gradient computation per batch |
| `spd/harvest/lib/harvester.py` | Accumulates attributions across batches |
| `spd/harvest/harvest.py` | Integration into harvest pipeline |
| `spd/harvest/storage.py` | `GlobalAttributionStorage` class (sparse COO format) |
| `spd/harvest/analysis.py` | Query functions: `get_top_attribution_sources/targets` |
| `spd/harvest/loaders.py` | `load_global_attributions()` function |
| `spd/harvest/scripts/run_harvest.py` | CLI flag `--compute_global_attributions` |
| `spd/app/backend/routers/correlations.py` | API endpoint for global attributions |
| `spd/app/backend/state.py` | Load global attributions into `HarvestCache` |
| `spd/app/frontend/src/lib/api/correlations.ts` | Frontend API client |
| `spd/app/frontend/src/components/ui/GlobalAttributionSection.svelte` | Display component |
| `spd/app/frontend/src/components/ui/GlobalAttributionsList.svelte` | List component |
| `spd/app/frontend/src/components/local-attr/ComponentNodeCard.svelte` | Integration into component details |
| `spd/app/frontend/src/lib/useComponentData.svelte.ts` | Fetch global attributions on component hover |

### Data Flow

```
Harvest Pipeline (offline, GPU):
  1. For each batch:
     a. Forward pass with unmasked masks + weight deltas
     b. Find "alive" components (CI > threshold)
     c. For each alive target: compute gradients w.r.t. all source layers
     d. Attribution = grad * activation (same sequence position)
  2. Accumulate into dense matrix (on GPU)
  3. Convert to sparse COO format on save

App (runtime):
  1. Load GlobalAttributionStorage from disk (lazy, on first request)
  2. Query top sources/targets for a given component
  3. Display in ComponentNodeCard and NodeTooltip
```

### Storage Format

Global attributions use sparse COO format:
- `indices`: `[2, nnz]` tensor - `indices[0]` = source indices, `indices[1]` = target indices
- `values`: `[nnz]` tensor - attribution values
- `component_keys`: list mapping indices to component keys like `"h.0.mlp.c_fc:5"`
- `n_components`: total component count
- `n_samples`: number of (batch * seq_len) samples processed

Saved to: `SPD_OUT_DIR/harvest/<run_id>/correlations/global_attributions.pt`

## Implementation Details

### `attribution_compute.py` - Core Algorithm

```python
def compute_batch_attributions(model, batch, ci_dict, ci_threshold, ...):
    # 1. Find alive components
    alive_mask = ci_flat > ci_threshold

    # 2. Setup unmasked forward pass with weight deltas
    weight_deltas = model.calc_weight_deltas()
    unmasked_masks = make_mask_infos(
        component_masks={...all ones...},
        weight_deltas_and_masks=weight_deltas_and_masks
    )

    # 3. Process targets in chunks (memory optimization)
    for chunk in chunks(all_alive_targets, FORWARD_CHUNK_SIZE):
        # Fresh forward pass per chunk
        comp_output = model(batch, mask_infos=unmasked_masks, ...)

        for target in chunk:
            # Compute gradient of target w.r.t. source activations
            grads = torch.autograd.grad(target_val, source_activations, ...)

            # Attribution = grad * activation
            attribution = grad * source_activation

    # 4. Return sparse edges
    return source_indices, target_indices, values
```

### Memory Optimization (OOM Fix)

The naive approach caused OOM with ~5600 alive components. The fix:

1. Process targets in chunks of `FORWARD_CHUNK_SIZE = 50`
2. Each chunk gets a fresh forward pass (new computation graph)
3. Only `retain_graph` within a chunk, not across chunks
4. Memory bounded by chunk size, not total targets

### Accumulation (`harvester.py`)

```python
class Harvester:
    # Dense accumulator (converted to sparse on save)
    attribution_sums: Tensor  # [n_components, n_components]

    def add_batch_attributions(self, src_indices, tgt_indices, values):
        # Accumulate into dense matrix
        self.attribution_sums.index_put_((src_indices, tgt_indices), values, accumulate=True)

    def _attribution_sums_to_sparse(self):
        # Convert to COO format for storage
        nonzero_mask = self.attribution_sums != 0
        src_indices, tgt_indices = torch.where(nonzero_mask)
        values = self.attribution_sums[src_indices, tgt_indices]
        return torch.stack([src_indices, tgt_indices]), values
```

### Analysis Functions (`analysis.py`)

```python
def get_top_attribution_sources(storage, component_key, top_k):
    """Get top components that attribute TO the query component."""
    # Find edges where component is target
    # Return (top_positive, top_negative) lists

def get_top_attribution_targets(storage, component_key, top_k):
    """Get top components that this component attributes TO."""
    # Find edges where component is source
    # Return (top_positive, top_negative) lists
```

### Frontend Display

`GlobalAttributionSection.svelte` shows:
- **Top Sources**: Components that attribute TO this component (positive + negative)
- **Top Targets**: Components this component attributes TO (positive + negative)

Integrated into `ComponentNodeCard.svelte` (shown in Components tab and hover tooltips).

## Usage

### CLI

```bash
python -m spd.harvest.scripts.run_harvest <wandb_path> \
    --compute_global_attributions \
    --n_batches 1000 \
    --batch_size 256
```

### Programmatic

```python
from spd.harvest.loaders import load_global_attributions
from spd.harvest.analysis import get_top_attribution_sources, get_top_attribution_targets

storage = load_global_attributions("run_id")
if storage:
    top_pos_sources, top_neg_sources = get_top_attribution_sources(storage, "h.0.mlp.c_fc:5", top_k=10)
    top_pos_targets, top_neg_targets = get_top_attribution_targets(storage, "h.0.mlp.c_fc:5", top_k=10)
```

## Current Status

### What Works

- All code implemented and passes type checks
- Bug fixes committed:
  - Off-by-one layer offset fixed
  - OOM issue fixed via chunking
  - `retain_graph` memory leak fixed
  - Gradient computation for unconnected layers fixed
  - Validation added to `GlobalAttributionStorage.load()`
- Frontend components render correctly
- API endpoints functional

### Known Issues

**Performance is the main blocker.** For models with many components (~5600 alive per batch):

- 112 forward passes per batch (5600 / 50 chunk size)
- 5600 backward passes per batch
- Each batch takes 10+ minutes
- Full harvest would take days

### Not Yet Tested

- End-to-end flow with a successfully harvested model
- App display with real data

## Remaining Work

### Required Before Use

1. **Performance optimization** - Choose one or more:
   - Increase `FORWARD_CHUNK_SIZE` (trade memory for speed)
   - Add subsampling parameter to limit alive targets per batch
   - Use higher CI threshold to reduce alive components
   - Run on cluster as long-running batch job

2. **End-to-end testing** - Run full harvest on a smaller model, verify app display

### Nice to Have

- Batch gradient computation (process multiple targets in single backward pass)
- Parallel processing across GPUs for attribution computation
- Progress logging during attribution computation

## Branch Information

| Branch | Purpose |
|--------|---------|
| `feature/global-attributions` | Main feature branch (this PR) |

## Key Commits

| Commit | Description |
|--------|-------------|
| `6879d611` | Initial global attributions implementation |
| `35c4e358` | Fix critical bugs (layer offset, validation) |
| `fa42ee59` | Add CLI flag |
| `2baf547e` | Fix gradient computation for unconnected layers |
| `c132439b` | Fix OOM via chunked forward passes |

## Questions for Continuation

1. What performance trade-off is acceptable? (time vs memory vs accuracy)
2. Is subsampling alive targets acceptable, or must all be computed?
3. What's the target model size for this feature?
4. Is running on cluster for extended periods an option?

## Contact

Original implementation by Lee with Claude Code assistance. See conversation logs in `~/.claude/projects/-mnt-polished-lake-home-lee-spd/` for detailed discussion history.
