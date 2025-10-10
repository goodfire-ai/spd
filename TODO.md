# TODO: Cluster Coactivation Matrix Implementation

## What Was Changed

### 1. Added `ClusterActivations` dataclass (`spd/clustering/dashboard/compute_max_act.py`)
- New dataclass to hold vectorized cluster activations for all clusters
- Contains `activations` tensor [n_samples, n_clusters] and `cluster_indices` list

### 2. Added `compute_all_cluster_activations()` function
- Vectorized computation of all cluster activations at once
- Replaces the per-cluster loop for better performance
- Returns `ClusterActivations` object

### 3. Added `compute_cluster_coactivations()` function
- Computes coactivation matrix from list of `ClusterActivations` across batches
- Binarizes activations (acts > 0) and computes matrix multiplication: `activation_mask.T @ activation_mask`
- Follows the pattern from `spd/clustering/merge.py:69`
- Returns tuple of (coactivation_matrix, cluster_indices)

### 4. Modified `compute_max_activations()` function
- Now accumulates `ClusterActivations` from each batch in `all_cluster_activations` list
- Calls `compute_cluster_coactivations()` to compute the matrix
- **Changed return type**: now returns `tuple[DashboardData, np.ndarray, list[int]]`
  - Added coactivation matrix and cluster_indices to return value

### 5. Modified `spd/clustering/dashboard/run.py`
- Updated to handle new return value from `compute_max_activations()`
- Saves coactivation matrix as `coactivations.npz` in the dashboard output directory
- NPZ file contains:
  - `coactivations`: the [n_clusters, n_clusters] matrix
  - `cluster_indices`: array mapping matrix positions to cluster IDs

## What Needs to be Checked

### Testing
- [ ] **Run the dashboard pipeline** on a real clustering run to verify:
  - Coactivation computation doesn't crash
  - Coactivations are saved correctly to NPZ file
  - Matrix dimensions are correct
  - `cluster_indices` mapping is correct

### Type Checking
- [ ] Run `make type` to ensure no type errors were introduced
- [ ] Verify jaxtyping annotations are correct

### Verification
- [ ] Load a saved `coactivations.npz` file and verify:
  ```python
  data = np.load("coactivations.npz")
  coact = data["coactivations"]
  cluster_indices = data["cluster_indices"]
  # Check: coact should be symmetric
  # Check: diagonal should be >= off-diagonal (clusters coactivate with themselves most)
  # Check: cluster_indices length should match coact.shape[0]
  ```

### Performance
- [ ] Check if vectorization actually improved performance
- [ ] Monitor memory usage with large numbers of clusters

### Edge Cases
- [ ] Test with clusters that have zero activations
- [ ] Test with single-batch runs
- [ ] Test with very large number of clusters

### Integration
- [ ] Verify the coactivation matrix can be used in downstream analysis
- [ ] Consider if visualization of coactivations should be added to dashboard

## Notes
- The coactivation matrix is computed over all samples processed (n_batches * batch_size * seq_len samples)
- Binarization threshold is currently hardcoded as `> 0` - may want to make this configurable
- The computation happens in the dashboard pipeline, NOT during the main clustering pipeline
