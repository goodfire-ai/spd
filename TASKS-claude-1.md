# Task 1: Batch Storage Infrastructure - Completion Report

## Summary

Completed Task 1 from `TODO-multibatch.md`: Batch Storage Infrastructure. This provides the foundation for multi-batch clustering by implementing data structures and precomputation logic for activation batches.

## Changes Made

### 1. Created `spd/clustering/batched_activations.py` (~180 lines)

This file consolidates all batch-related functionality (previously split between `batched_activations.py` and `precompute_batches.py`).

**Components:**
- **`ActivationBatch`**: Dataclass for storing a single batch of activations with labels
  - `save()`: Saves batch to disk as `.pt` file
  - `load()`: Loads batch from disk

- **`BatchedActivations`**: Iterator for cycling through multiple batches from disk
  - Finds all `batch_*.pt` files in a directory
  - `get_next_batch()`: Returns next batch in round-robin fashion
  - `n_batches`: Property for total number of batches

- **`precompute_batches_for_ensemble()`**: Function to generate all batches for ensemble runs
  - Loads model once, generates all batches for all runs
  - Saves batches to disk with structure: `<output_dir>/precomputed_batches/run_{idx}/batch_{idx}.pt`
  - Returns `None` if `recompute_costs_every=1` (single-batch mode)
  - Uses unique seeds per batch: `base_seed + run_idx * 1000 + batch_idx`

### 2. Updated `spd/clustering/merge_config.py`

**Changed:**
- `recompute_costs_every`: Updated default from `10` to `1` (original behavior)
- Updated description to match TODO spec

**Rationale:** Default of 1 maintains backward compatibility - single batch mode is the original behavior.

### 3. Updated `spd/clustering/clustering_run_config.py`

**Added:**
- `precomputed_activations_dir: Path | None = None`
- Description: "Path to directory containing precomputed activation batches. If None, batches will be auto-generated before merging starts."

**Key Design Decision:** When `None`, the system will auto-generate all required batches in a temp directory before merge starts (Option A from user clarification).

### 4. Updated `spd/clustering/scripts/run_clustering.py`

**Added:**
- CLI argument: `--precomputed-activations-dir`
- Override logic to pass this value to the config

### 5. Merged Files

**Deleted:** `spd/clustering/precompute_batches.py`

**Rationale:** The two files were tightly coupled (~50 and ~140 lines), used together, and would evolve together. Combining them into one `batched_activations.py` file makes the codebase simpler.

**Updated imports:**
- `spd/clustering/scripts/run_pipeline.py`: Now imports from `batched_activations`

## Concerns & Notes

### 1. **Circular Import Risk** ⚠️

In `batched_activations.py`, the `precompute_batches_for_ensemble()` function has a type annotation:
```python
def precompute_batches_for_ensemble(
    clustering_run_config: "ClusteringRunConfig",  # String annotation to avoid circular import
    ...
)
```

This is a forward reference (string annotation) because:
- `batched_activations.py` imports from `clustering` modules
- `clustering_run_config.py` might import from `batched_activations.py` in the future

**Status:** Currently safe, but watch for circular imports if we add imports to configs.

### 2. **Type Checking Not Verified** ⚠️

I attempted to run a basic import test but the user interrupted. We should verify:
- `basedpyright` passes
- No circular import errors at runtime
- All imports resolve correctly

**Recommended:** Run `make check` to verify type safety.

### 3. **Config Default Behavior Change**

The default for `recompute_costs_every` was changed from `10` → `1`. This affects any existing configs that relied on the implicit default of 10.

**Impact:** Likely minimal since this appears to be new functionality, but worth noting for any in-progress experiments.

### 4. **Seeding Strategy**

The seed calculation for batches is:
```python
seed = base_seed + run_idx * 1000 + batch_idx
```

**Assumption:** Maximum of 1000 batches per run. If more than 1000 batches needed, seeds could collide across runs.

**Mitigation:** Very unlikely - 1000 batches would require either:
- Very long merge iterations, or
- Very small `recompute_costs_every` values

### 5. **Disk Space Considerations**

Batches are saved to disk with activations on CPU. For large ensembles:
- `n_runs * n_batches_per_run * batch_size_on_disk`
- Could be substantial for large models/datasets

**Note:** No cleanup mechanism implemented - batches persist after runs complete.

### 6. **TODO Document Discrepancy**

The TODO document (lines 418-483) describes behavior where:
- `precomputed_activations_dir=None` → "compute single batch on-the-fly"
- `precomputed_activations_dir=<path>` → "use precomputed batches"

**Actual Implementation (per user request):**
- `precomputed_activations_dir=None` → "auto-generate all batches, then run merge"
- `precomputed_activations_dir=<path>` → "use precomputed batches"

This follows "Option A" clarified by the user. The TODO document may need updating.

## Next Steps

### Immediate
1. **Verify type checking:** Run `make check` or `basedpyright`
2. **Test imports:** Ensure no circular import issues
3. **Update TODO-multibatch.md:** Reflect the actual implementation of Option A

### For Task 2 (Core Merge Logic Refactor)
The infrastructure is ready:
- `BatchedActivations` can be used in `merge_iteration()`
- `recompute_coacts_from_scratch()` needs to be added
- NaN masking logic needs to be implemented
- Merge pair samplers need NaN handling

## Files Modified

```
Created:
- spd/clustering/batched_activations.py (new, ~180 lines)

Modified:
- spd/clustering/merge_config.py (updated default + description)
- spd/clustering/clustering_run_config.py (added 1 field)
- spd/clustering/scripts/run_clustering.py (added CLI arg + override)
- spd/clustering/scripts/run_pipeline.py (updated import)

Deleted:
- spd/clustering/precompute_batches.py (merged into batched_activations.py)
```

## Testing Recommendations

1. **Unit tests** (create `tests/clustering/test_batched_activations.py`):
   - Test `ActivationBatch.save()` and `load()`
   - Test `BatchedActivations` cycling behavior
   - Test that `n_batches` property works correctly

2. **Integration test**:
   - Run `precompute_batches_for_ensemble()` with small config
   - Verify batch files are created with correct naming
   - Verify `BatchedActivations` can load and cycle through them

3. **Backward compatibility test**:
   - Run with `recompute_costs_every=1` (should behave as before)
   - Verify no batches are precomputed when not needed
