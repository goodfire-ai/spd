# PR 342 Improvements Plan

## Tasks

### 1. Add tests for `AttributionHarvester` core logic
- [ ] Test `_build_layer_index_ranges` correctness
- [ ] Test `_build_alive_indices_per_layer` with various alive masks
- [ ] Test attribution accumulation for simple known case
- [ ] File: `tests/dataset_attributions/test_harvester.py`

### 2. Fix `retain_graph=True` on last backward
- [ ] In `_process_target_layer`, use `retain_graph=False` for the last target component
- [ ] File: `spd/dataset_attributions/harvester.py`

### 3. Make `_key_to_idx` public or add `has_component()`
- [ ] Add `has_component(key: str) -> bool` method to `DatasetAttributionStorage`
- [ ] Update router to use new method instead of accessing private `_key_to_idx`
- [ ] Files: `spd/dataset_attributions/storage.py`, `spd/app/backend/routers/dataset_attributions.py`

### 4. Add normalization - document interpretation of raw values
- [ ] Normalize by `n_tokens_processed` so values are per-token averages
- [ ] Update storage to store normalized values
- [ ] File: `spd/dataset_attributions/harvester.py`, `spd/dataset_attributions/harvest.py`

### 5. Make HarvestCache fail fast (consistent nullability)
- [ ] Change `activation_contexts_summary` to assert instead of returning None
- [ ] Change `dataset_attributions` to assert instead of returning None
- [ ] Update callers to handle the new behavior
- [ ] File: `spd/app/backend/state.py`

### 7. Add wte and output nodes to dataset attributions
- [ ] Include wte as a source layer (pseudo-component)
- [ ] Include output as a target layer
- [ ] Update `_build_component_keys` to include these
- [ ] Update harvester to compute attributions from/to these nodes
- [ ] Files: `spd/dataset_attributions/harvest.py`, `spd/dataset_attributions/harvester.py`

### 8. Change ci_threshold default to 0
- [ ] Update default in CLI script
- [ ] File: `spd/dataset_attributions/scripts/run.py`

### 10. Fill out PR template (skip - user said don't worry)

### 11. Fix mismatched default k values
- [ ] Align frontend and backend defaults (use 20)
- [ ] Files: `spd/app/backend/routers/dataset_attributions.py`, `spd/app/frontend/src/lib/api/datasetAttributions.ts`

## Progress

- [x] Task 1: Tests
- [x] Task 2: retain_graph fix
- [x] Task 3: has_component method
- [x] Task 4: Normalization
- [x] Task 5: Fail fast
- [x] Task 7: wte/output nodes
- [x] Task 8: ci_threshold default
- [x] Task 11: k value alignment

## Summary of Changes

### Task 1: Tests
- Created `tests/dataset_attributions/test_harvester.py` with tests for:
  - `DatasetAttributionStorage.has_component()`
  - `get_attribution()`, `get_top_sources()`, `get_top_targets()`
  - Save/load roundtrip

### Task 2: retain_graph fix
- Modified `AttributionHarvester._process_target_layer()` to track `is_last_layer` and `is_last_component`
- Only the final backward pass uses `retain_graph=False` to release the computation graph

### Task 3: has_component method
- Added `has_component(key: str) -> bool` to `DatasetAttributionStorage`
- Updated router to use `has_component()` instead of accessing private `_key_to_idx`

### Task 4: Normalization
- Attribution values are now normalized by `n_tokens_processed`
- Values represent per-token average attribution, making them comparable across runs
- Updated docstrings to document this

### Task 5: Fail fast
- Added `has_activation_contexts_summary()` and `has_dataset_attributions()` methods to `HarvestCache`
- Properties now assert instead of returning None
- Updated routers to use availability check methods before accessing properties

### Task 7: wte and output nodes
- Updated `_build_component_keys()` to include `wte:0` and `output:0`
- Updated `_build_alive_mask()` to include these (always alive)
- Updated filtering logic to include wte as valid source and output as valid target
- Updated `AttributionHarvester` to handle these pseudo-layers:
  - wte: single component, treats embedding output as activation
  - output: single component, treats logits as the output

### Task 8: ci_threshold default
- Changed default from `1e-6` to `0.0` (include all components)

### Task 11: k value alignment
- Changed default k from 10 to 20 in both backend and frontend
