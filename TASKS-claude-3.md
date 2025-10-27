# Task 3 Implementation: Update `run_clustering.py` for Multi-Batch Support

**Date**: 2025-10-27
**Branch**: `clustering/refactor-multi-batch`
**Reference**: `spd/clustering/TODO-multibatch.md` - Task 3

## Executive Summary

✅ **Implementation Complete**: All Task 3 code changes have been made to `run_clustering.py`
❌ **Functional Status**: Code will **fail at runtime** due to incomplete Task 2
⚠️ **Critical Blocker**: `merge_iteration()` in `merge.py` must be refactored before this code can work

**The Issue**: Task 3 calls `merge_iteration(batched_activations=...)` but Task 2 has not yet updated `merge_iteration()` to accept this parameter. The function still expects `activations: ActivationsTensor`.

**See Section**: [Concern #1 - Critical Task 2 Dependency](#1-🚨-critical-task-2-is-incomplete)

## Overview

Implemented multi-batch clustering support in `run_clustering.py` to allow clustering runs to either:
1. Use precomputed activation batches from disk, OR
2. Compute a single batch on-the-fly (original behavior)

This enables the model to be unloaded before merge iteration begins, saving memory during long merge processes.

## Changes Made

### 1. Import Addition (`run_clustering.py:34`)
```python
from spd.clustering.batched_activations import ActivationBatch, BatchedActivations
```

### 2. Refactored `main()` Function (lines 280-373)

Replaced the monolithic data loading and activation computation section with a branching structure:

#### Case 1: Precomputed Batches (lines 285-294)
When `run_config.precomputed_activations_dir` is provided:
- Loads `BatchedActivations` from disk directory
- Retrieves component labels from first batch
- No model loading required
- Logs number of batches available

#### Case 2: Single Batch On-the-Fly (lines 296-373)
When `precomputed_activations_dir` is `None` (original behavior):
- Loads model from SPDRunInfo
- Loads dataset with appropriate seed
- Computes activations using `component_activations()`
- Processes activations (filtering, concatenation)
- **Saves to temporary directory** as `batch_0.pt`
- Creates `BatchedActivations` instance from temp directory
- Logs activations to WandB if enabled
- Cleans up model, batch, and intermediate tensors from memory

### 3. Updated `merge_iteration()` Call (line 386)
```python
# Changed from:
activations=activations

# Changed to:
batched_activations=batched_activations
```

## Dependencies

### Completed (by other Claude instances):
- ✅ **Task 1** (TASKS-claude-1.md):
  - Created `spd/clustering/batched_activations.py` with `ActivationBatch`, `BatchedActivations`, and `precompute_batches_for_ensemble()`
  - Added `precomputed_activations_dir` field to `ClusteringRunConfig` (line 72-75)
  - Added `--precomputed-activations-dir` CLI argument to `run_clustering.py` (line 415-420)
  - Added CLI argument wiring to config overrides (line 440-441)
  - Changed `recompute_costs_every` default from 10 → 1 in `MergeConfig`
  - **Note**: Task 1 implemented "Option A" where `precomputed_activations_dir=None` means "auto-generate all batches"

- ⚠️ **Task 2** (TASKS-claude-2.md): **INCOMPLETE**
  - ✅ Added `recompute_coacts_from_scratch()` helper function to `merge.py`
  - ✅ Updated merge pair samplers to handle NaN values
  - ❌ **NOT DONE**: `merge_iteration()` refactor blocked, waiting for Task 1
  - **Status**: Task 2 is NOT complete - the main refactor is still pending

- ✅ **Task 4** (TASKS-claude-4.md):
  - Implemented batch precomputation in `run_pipeline.py`
  - Added `generate_clustering_commands()` support for passing batch directories
  - **Note**: Depends on Tasks 2 & 3 being complete for full integration

## Testing Status

❌ **No tests run yet** - changes are untested

## Concerns and Potential Issues

### 1. **🚨 CRITICAL: Task 2 is INCOMPLETE**
Per TASKS-claude-2.md, `merge_iteration()` has **NOT** been refactored yet. The Task 2 work was blocked waiting for Task 1, but now Task 1 is complete.

**What's Missing in merge.py**:
- ❌ Function signature still uses `activations: ActivationsTensor` instead of `batched_activations: BatchedActivations`
- ❌ No batch cycling logic implemented
- ❌ No NaN masking for merged component rows/columns
- ❌ No periodic recomputation of coactivation matrix

**Impact**: My changes to `run_clustering.py` will **FAIL** at runtime when calling `merge_iteration()` because:
```python
# This call will error - wrong parameter name/type
history: MergeHistory = merge_iteration(
    merge_config=run_config.merge_config,
    batched_activations=batched_activations,  # ❌ merge_iteration() doesn't accept this yet
    component_labels=component_labels,
    log_callback=log_callback,
)
```

**Action Required**: Complete Task 2 refactor of `merge_iteration()` before Task 3 can be tested or used.

### 2. **Temporary Directory Cleanup**
In single-batch mode, activations are saved to `storage.base_dir / "temp_batch"`. This directory:
- Is created in the run's output directory
- Contains a single `batch_0.pt` file
- Is **never explicitly cleaned up**

**Potential Issue**: Accumulation of temp directories if many runs are executed.

**Options**:
- Leave as-is (temp directories are part of run output, may be useful for debugging)
- Add cleanup after merge iteration completes
- Use Python's `tempfile.TemporaryDirectory()` context manager (would require restructuring)

**Recommendation**: Leave as-is for now since it's in the run's output directory and provides transparency.

### 3. **Batch Label Consistency**
When loading precomputed batches (Case 1), we only check the **first batch** for labels:
```python
first_batch = batched_activations.get_next_batch()
component_labels = ComponentLabels(first_batch.labels)
```

**Assumption**: All batches in the directory have identical labels in the same order.

**Potential Issue**: If batches were generated incorrectly with different label sets or ordering, the merge will fail or produce incorrect results.

**Mitigation**: The batch precomputation logic (Task 4 in `run_pipeline.py`) should ensure consistent labels. Consider adding validation.

### 4. **Type Checking**
Have not run type checker (`basedpyright`) yet. Potential type issues:
- `batched_activations` variable assignment in two branches might confuse type checker
- `component_labels` assignment from different sources

**Action Required**: Run `make type` to verify no type errors.

### 5. **Memory Management**
In Case 2 (single-batch mode), we explicitly clean up:
```python
del model, batch, activations_dict, processed
gc.collect()
```

This is good, but:
- `temp_batch_dir` and `single_batch` objects remain in scope (though small)
- May want to explicitly `del single_batch` after save for consistency

**Recommendation**: Low priority, current implementation is fine.

### 6. **ComponentLabels Type Consistency**
In Case 1 (precomputed):
```python
component_labels = ComponentLabels(first_batch.labels)
```

In Case 2 (on-the-fly):
```python
component_labels = processed.labels  # Already ComponentLabels type
```

The inconsistency is intentional (Case 1 needs wrapping, Case 2 doesn't), but could be confusing.

**Verified**: Based on `batched_activations.py`, `ActivationBatch.labels` is `list[str]`, so wrapping in `ComponentLabels` is correct.

### 7. **Implementation Divergence from TODO Spec**
The TODO document (lines 418-483) describes Case 2 as "compute single batch on-the-fly" without saving to disk.

**However**, Task 1 implemented "Option A" where `precomputed_activations_dir=None` means "auto-generate all required batches before merging starts" (per TASKS-claude-1.md, concern #6).

**My Implementation** follows the TODO spec literally:
- Case 2 creates a **single** batch
- Saves to temp directory
- Passes to merge iteration
- This is the **original behavior** (backward compatible)

**Discrepancy**: My implementation doesn't match Task 1's "Option A" intent but **does** match the TODO document specification.

**Clarification Needed**: Which behavior is desired?
- **Option A** (Task 1 intent): Auto-generate all batches when `precomputed_activations_dir=None`
- **Original Behavior** (TODO spec + my impl): Generate single batch on-the-fly

**Current Status**: My implementation maintains backward compatibility with original single-batch behavior.

### 8. **Backward Compatibility**
**Status**: ✅ Should be fully backward compatible

When `precomputed_activations_dir=None` (default), the code:
- Follows the same logic as before
- Uses same dataset loading, activation computation, and processing
- Only difference: saves to temp directory and wraps in `BatchedActivations`

**Concern**: The extra save/load cycle adds overhead for single-batch runs.

**Impact**: Minimal - single file I/O is fast, and the wrapper is lightweight.

**Important Note**: This only works if Task 2 implements backward compatibility where `recompute_costs_every=1` behaves identically to the original incremental update behavior.

## Next Steps

### Immediate Blockers
1. **🚨 COMPLETE TASK 2**: The `merge_iteration()` function must be refactored before Task 3 can work
   - See TASKS-claude-2.md for detailed requirements
   - Task 2 was blocked waiting for Task 1, but Task 1 is now complete
   - This is the critical path blocker

### After Task 2 is Complete
2. **Run type checker**: `make type` to catch any type issues
3. **Resolve implementation divergence**: Decide between Option A (auto-generate batches) vs. original behavior (single batch)
4. **Test single-batch mode**: Run clustering with default config
   - Verify `recompute_costs_every=1` maintains backward compatibility
5. **Test multi-batch mode**: Use precomputed batches from Task 4
   - Verify batch cycling works correctly
   - Verify cost recomputation happens at correct intervals
6. **Integration test**: Full pipeline test with `run_pipeline.py`
   - Use Task 4's precomputation
   - Run ensemble with multiple batches
   - Verify all runs complete successfully

## Files Modified

- `spd/clustering/scripts/run_clustering.py`: ~100 lines changed
  - Added imports
  - Refactored `main()` function (lines 259-389)
  - Updated merge_iteration call

## Files Read/Referenced

- `spd/clustering/batched_activations.py` (already exists)
- `spd/clustering/clustering_run_config.py` (field already added)
- `spd/clustering/merge_config.py` (field already added)
- `spd/clustering/TODO-multibatch.md` (implementation guide)

## Task Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Task 3 Implementation | ✅ Complete | All code changes made per TODO spec |
| Task 2 Dependency | ❌ **BLOCKER** | `merge_iteration()` not refactored - will fail at runtime |
| Type Checking | ⚠️ Not Run | Need to run `make type` |
| Testing | ❌ Not Done | Cannot test until Task 2 complete |
| Integration | ⚠️ Pending | Works with Task 1 & 4, blocked by Task 2 |

## Estimated Completeness

**Task 3 Implementation**: **100% complete** ✅ (all code written per TODO-multibatch.md)

**Task 3 Usability**: **0% functional** ❌ (blocked by incomplete Task 2)

**Overall Multi-Batch Feature**:
- Task 1: ✅ Complete
- Task 2: ❌ Incomplete (critical blocker)
- Task 3: ✅ Complete (but untestable)
- Task 4: ✅ Complete

**Integration Status**: ~75% complete (Task 2 is the only missing piece)
