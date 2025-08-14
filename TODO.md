# WandB Integration for SPD Clustering

## Metrics to Log at Each Iteration

### Core Statistics (Already Identified)
- **Coactivation matrix stats**: min, max, mean, median, std, quantiles
- **Cost matrix stats**: min, max, mean, median, std, quantiles  
- **Group activations**: Mean/max activation per group
- **Group size distribution**: Number of groups of each size (histogram data)

### Additional Metrics to Consider
- **Merge selection metrics**:
  - Selected pair indices and cost
  - Rank of selected pair in cost matrix
  - Gap between best and second-best costs
- **Convergence metrics**:
  - Change in mean/median costs from previous iteration
  - Rate of cost decrease
- **Sparsity metrics**:
  - Percentage of zero entries in coactivation matrix
  - Average coactivation density per group
- **Component tracking**:
  - Number of singleton groups
  - Largest group size
  - Group size variance
- **Computational metrics** (optional):
  - Iteration time
  - Memory usage

## Artifact Logging Strategy

### GroupMerge Checkpoints
- Save BatchedGroupMerge artifacts every N iterations (configurable, default 100)
- Artifact should contain:
  - `group_idxs`: Int tensor of shape `(n_iters_saved, n_components)`
  - `k_groups`: Number of groups at each saved iteration
  - `selected_pairs`: The merge pairs selected at each iteration
  - `costs`: Cost of selected pairs
  - Metadata: iteration numbers, timestamp, config hash

## Implementation Plan

### 1. Configuration Changes

#### [spd/clustering/merge_run_config.py](spd/clustering/merge_run_config.py)
- Add WandB configuration fields:
  - `wandb_enabled: bool = False`
  - `wandb_project: str = "spd-cluster"`
  - `wandb_log_frequency: int = 1` (log every N iterations)
  - `wandb_artifact_frequency: int = 100` (save artifacts every N iterations)
- Extract parent model WandB run ID from `model_path`

#### [spd/clustering/merge_config.py](spd/clustering/merge_config.py)
- Add fields to pass WandB config down to merge iteration

### 2. Main Pipeline Changes

#### [spd/clustering/scripts/main.py](spd/clustering/scripts/main.py)
- Initialize parent WandB run for the ensemble
- Extract parent model run ID from `model_path`
- Create WandB group based on parent model
- Log ensemble-level configuration
- After completion, aggregate and log ensemble statistics

### 3. Individual Batch Processing

#### [spd/clustering/scripts/s2_run_clustering.py](spd/clustering/scripts/s2_run_clustering.py)
- Initialize child WandB run for each batch
- Use same group as parent (model-based grouping)
- Pass WandB run object to `merge_iteration`
- Log final plots and summaries to WandB

### 4. Core Merge Logic

#### [spd/clustering/merge.py](spd/clustering/merge.py)
- Add `wandb_run` parameter to `merge_iteration`
- Compute additional metrics at each iteration:
  ```python
  # After line 137 (merge_history.add_iteration)
  if wandb_run is not None:
      # Log all metrics
  ```
- Implement artifact saving logic:
  ```python
  if wandb_run and i % config.wandb_artifact_frequency == 0:
      # Save BatchedGroupMerge as artifact
  ```

### 5. Metric Computation Utilities

#### [spd/clustering/merge_history.py](spd/clustering/merge_history.py)
- Add method to compute group size distribution
- Add method to compute activation statistics per group
- Add method to export data for WandB artifacts

### 6. WandB Utilities

#### [spd/utils/wandb_utils.py](spd/utils/wandb_utils.py)
- Add helper function to parse model path and extract WandB run ID
- Add helper to create consistent group names
- Add artifact creation utilities for BatchedGroupMerge

## Testing Strategy

1. Test with `wandb_enabled: false` to ensure no regression
2. Test with small toy example (few iterations) to verify logging
3. Test artifact saving and loading
4. Test ensemble aggregation with multiple batches

## Migration Path

1. All changes should be backward compatible
2. Default to `wandb_enabled: false` 
3. Existing scripts continue to work without modification
4. WandB features are opt-in via config

## Example Config Usage

```yaml
# In merge_run_config.yaml
wandb_enabled: true
wandb_project: "spd-cluster"
wandb_log_frequency: 1
wandb_artifact_frequency: 100
model_path: "wandb:my-entity/my-project/abc123"  # Links to parent SPD model
```

## Progress Update

### ✅ Completed
1. **Step 1: Configuration Changes** - Implemented WandB configuration in `MergeRunConfig`
   - Added `wandb_enabled`, `wandb_project`, `wandb_log_frequency`, `wandb_artifact_frequency` fields
   - Created `wandb_group` and `config_identifier` properties for proper organization
   - Added `from_experiment_key()` factory method
   - Updated config files to use new format
   - Handles both legacy `spd_exp:` format and new `experiment_key` approach

2. **WandB Tensor Logging Utilities** - Created `spd/utils/wandb_tensor_info.py`
   - `wandb_log_tensor()`: Uses muutils for console output + creates histogram plots for WandB
   - `wandb_log_dict()`: Uses muutils dbg_auto + logs structured data to WandB  
   - `wandb_log_figure()`: Logs matplotlib figures/images to WandB
   - Functions return original values (like dbg_tensor/dbg_auto) for drop-in replacement

### 🔄 In Progress
3. **Step 2: Initialize WandB in clustering pipeline**
   - Need to add WandB initialization in `s2_run_clustering.py` for batch runs
   - Need to add parent WandB run in `main.py` for ensemble coordination

### 📋 Next Steps
4. Replace `dbg_auto`/`dbg_tensor` calls with WandB equivalents in clustering code
5. Add WandB logging to `merge_iteration` in `merge.py`
6. Upload figures to WandB instead of just saving locally
7. Implement artifact saving for GroupMerge checkpoints
8. Test end-to-end with small example

## Implementation Notes

- WandB tensor histograms now use actual matplotlib plots with mean/median/stddev marked
- Console output still uses muutils for consistency
- All WandB features are opt-in via `wandb_enabled: false` default
- Config system handles both direct WandB paths and experiment registry keys