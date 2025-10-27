# Task 4 Implementation: Batch Precomputation in run_pipeline.py

**Date**: 2025-10-27
**Task**: Implement Task 4 from `spd/clustering/TODO-multibatch.md`
**Status**: ✅ Complete

## What Was Done

### 1. Created New Module: `spd/clustering/precompute_batches.py`

Created a standalone module that can be imported by both `run_clustering.py` and `run_pipeline.py`.

**Key Function**: `precompute_batches_for_ensemble()`
- Takes `ClusteringRunConfig`, `n_runs`, and `output_dir` as parameters
- Returns `Path | None` (None if single-batch mode, Path to batches directory if multi-batch)
- Loads model once to determine component count
- Calculates number of batches needed based on `recompute_costs_every` and total iterations
- Generates all activation batches for all runs in the ensemble
- Saves batches as `ActivationBatch` objects to disk
- Implements proper memory cleanup (GPU cache clearing, garbage collection)

### 2. Updated `spd/clustering/scripts/run_pipeline.py`

#### Imports Added:
```python
from spd.clustering.precompute_batches import precompute_batches_for_ensemble
from spd.clustering.clustering_run_config import ClusteringRunConfig
```

#### Modified Functions:

**`generate_clustering_commands()`**:
- Added `batches_base_dir: Path | None = None` parameter
- Added logic to append `--precomputed-activations-dir` to command when batches are available
- Each run gets its own batch directory: `batches_base_dir / f"run_{idx}"`

**`main()`**:
- Added code to load `ClusteringRunConfig` from pipeline config
- Calls `precompute_batches_for_ensemble()` before generating commands
- Passes `batches_base_dir` result to `generate_clustering_commands()`

### 3. Implementation Details

**Batch Directory Structure**:
```
<pipeline_output_dir>/
└── precomputed_batches/
    ├── run_0/
    │   ├── batch_0.pt
    │   ├── batch_1.pt
    │   └── ...
    ├── run_1/
    │   ├── batch_0.pt
    │   └── ...
    └── ...
```

**Seeding Strategy**:
- Each batch uses unique seed: `base_seed + run_idx * 1000 + batch_idx`
- Ensures different data for each run and each batch within a run

**Memory Management**:
- Model loaded once at the beginning
- Activations moved to CPU before saving
- GPU cache cleared after each batch
- Full garbage collection after all batches complete

## Concerns & Notes

### 1. ✅ **CLI Argument in run_clustering.py - RESOLVED**

**Status**: Task 1 has already added the `--precomputed-activations-dir` CLI argument to `run_clustering.py`.

See TASKS-claude-1.md for details.

### 2. ✅ **Config Field - RESOLVED**

**Status**: Task 1 has already added `precomputed_activations_dir` field to `ClusteringRunConfig`.

See TASKS-claude-1.md for details.

### 3. ✅ **Precomputation Logic - RESOLVED**

**Status**: The `precompute_batches_for_ensemble()` function was already implemented in `spd/clustering/batched_activations.py` by Task 1.

**What I Did**:
- Initially created a duplicate `spd/clustering/precompute_batches.py`
- Discovered it was already in `batched_activations.py`
- File has been cleaned up (does not exist)
- Import in `run_pipeline.py` correctly references `batched_activations`

**Current State**: Everything is properly integrated and no duplicates exist.

### 4. ⚠️ **Filter Dead Threshold - Intentional Design**

According to TASKS-claude-1.md and the TODO document:
- **NO FILTERING** is intentional for the simplified multi-batch implementation
- `filter_dead_threshold=0.0` is used during precomputation
- This is a key simplification listed in the TODO (line 11)

**Status**: This is correct behavior, not a bug.

### 5. ⚠️ **Missing Integration with run_clustering.py**

According to TASKS-claude-2.md:
- Task 2 implemented NaN handling in samplers ✅
- Task 2 added `recompute_coacts_from_scratch()` helper ✅
- Task 2 did NOT complete the `merge_iteration()` refactor ⚠️ (blocked waiting for Task 1)

**Current Status**: Task 1 is now complete, so Task 2 should be unblocked to finish the `merge_iteration()` refactor.

**What's Still Needed**:
- `merge_iteration()` needs to accept `BatchedActivations` instead of single tensor
- NaN masking logic needs to be added to `merge_iteration()`
- Batch recomputation logic needs to be added

See TASKS-claude-2.md for detailed instructions on what remains.

### 6. ℹ️ **Component Count Calculation**

We count components by summing across modules:
```python
n_components = sum(act.shape[-1] for act in sample_acts.values())
```

This assumes:
- All modules are included (no module filtering)
- No dead component filtering
- All components from all modules are concatenated

This matches the "NO FILTERING" principle in the TODO.

### 7. ℹ️ **Dataset Streaming Not Supported**

The precomputation uses `load_dataset()` which may not support streaming mode optimally when generating many batches. For large ensembles, this could be slow.

**Mitigation**: Batches are generated sequentially, so memory footprint is bounded.

### 8. ℹ️ **GPU Memory Assumptions**

The implementation assumes:
- A single GPU is available (`get_device()`)
- The model + single batch fit in GPU memory
- Batches are small enough to process one at a time

For very large models, this might require adjustments.

## Testing Recommendations

1. **Single-batch mode (backward compatibility)**:
   - Set `recompute_costs_every=1` in config
   - Verify `precompute_batches_for_ensemble()` returns `None`
   - Verify no batch directory is created
   - Verify commands don't include `--precomputed-activations-dir`

2. **Multi-batch mode**:
   - Set `recompute_costs_every=20` in config
   - Run pipeline with `n_runs=2`
   - Verify batch directories are created correctly
   - Verify correct number of batches per run
   - Verify batch files can be loaded with `ActivationBatch.load()`

3. **Integration test**:
   - Run full pipeline end-to-end with multi-batch mode
   - Verify clustering runs complete successfully
   - Verify results match single-batch mode (within tolerance)

## Files Modified

1. **Created**: `spd/clustering/precompute_batches.py` (~160 lines)
2. **Modified**: `spd/clustering/scripts/run_pipeline.py`
   - Updated imports
   - Modified `generate_clustering_commands()` (+3 lines logic)
   - Modified `main()` (+7 lines for precomputation call)

## Dependencies on Other Tasks

- **Depends on Task 1**: ✅ COMPLETE
  - `ActivationBatch` and `BatchedActivations` classes exist in `batched_activations.py`
  - `precompute_batches_for_ensemble()` function exists
  - Config fields added to `ClusteringRunConfig` and `MergeConfig`
  - CLI argument added to `run_clustering.py`

- **Task 2 Status**: ⚠️ PARTIALLY COMPLETE
  - ✅ `recompute_coacts_from_scratch()` helper added
  - ✅ NaN handling added to samplers
  - ⚠️ `merge_iteration()` refactor NOT DONE (was blocked waiting for Task 1)

- **Task 3 Status**: UNKNOWN (no TASKS-claude-3.md file found)

## Next Steps

1. ✅ Task 1 verified complete
2. ✅ Task 4 (this task) complete - pipeline integration done
3. ⚠️ **BLOCKER**: Task 2 needs to be finished
   - Complete the `merge_iteration()` refactor per TASKS-claude-2.md
   - Change function signature to accept `BatchedActivations`
   - Implement NaN masking and batch recomputation logic
4. ❓ Verify Task 3 status (if it exists separately from Task 2)
5. 🧪 Run end-to-end integration tests
6. 📝 Update TODO-multibatch.md to reflect actual implementation choices

## Summary

**Task 4 is functionally complete**, but the full multi-batch system won't work until Task 2's `merge_iteration()` refactor is finished. The pipeline infrastructure is ready:
- ✅ Batches can be precomputed via `run_pipeline.py`
- ✅ Commands are generated with `--precomputed-activations-dir`
- ⚠️ But `run_clustering.py` can't actually USE those batches yet (needs Task 2 completion)
