# Performance Optimization: `_store_activations` in `batch_storage.py`

## Current Bottleneck

The `_store_activations()` method is the slowest part of batch processing.

### Performance Issues:

1. **Triple nested loop**: `O(n_clusters � batch_size � n_components_per_cluster)`
   - Example: 100 clusters � 32 samples � 5 components = **16,000 iterations per batch**

2. **Repeated `.to_string()` calls** (line 258):
   - Called for every cluster, every batch
   - Should be pre-computed once in `.create()`

3. **Repeated dictionary lookups in hot loop** (lines 267-268, 285-288):
   - Same hash looked up multiple times per cluster

4. **GPU�CPU transfers inside innermost loop** (lines 280-282):
   - `.cpu().numpy()` called for EVERY component of EVERY sample
   - Example: 100 clusters � 32 samples � 5 components = **16,000 GPU�CPU transfers per batch**!
   - This is likely the #1 bottleneck

5. **Individual `.append()` calls instead of vectorized `.extend()`**:
   - List appends can cause reallocations

6. **Repeated `processed.get_label_index()` lookups** (line 275):
   - Called inside hot loop instead of pre-computed

7. **Repeated `self.cluster_components[cluster_idx]` lookup** (line 272):
   - Same lookup for every sample in the batch

---

## Optimization Ideas (Ranked by Impact)

### =% HIGH IMPACT (10-100x speedup potential)

#### 1. Move ALL GPU�CPU transfers outside loops
```python
# ONCE at the start of the function:
processed_acts_cpu = processed.activations.cpu().numpy()  # Shape: [batch*seq, n_components]
processed_acts_3d = processed_acts_cpu.reshape(batch_size, seq_len, -1)  # [batch, seq, n_comps]

# Then inside loops, just index the numpy array (no .cpu() calls!)
comp_acts_1d = processed_acts_3d[sample_idx, :, comp_idx]  # Pure numpy indexing, no GPU transfer!
```

#### 2. Vectorize storage - use `.extend()` instead of nested loop appends
```python
# Current: O(batch_size) appends per cluster
for sample_idx in range(batch_size):
    self.cluster_activations[hash].append(cluster_acts_2d[sample_idx])

# Optimized: Single extend with list of arrays
batch_acts_list = [cluster_acts_2d[i] for i in range(batch_size)]
self.cluster_activations[hash].extend(batch_acts_list)

# Or even better if numpy supports it:
self.cluster_activations[hash].extend(cluster_acts_2d)
```

#### 3. Pre-compute cluster_hash_map during `.create()`
```python
# In BatchProcessingStorage.create():
cluster_hash_map: dict[int, ClusterIdHash] = {
    idx: cluster_id_map[idx].to_string()
    for idx in unique_cluster_indices
}

# Store as attribute:
self.cluster_hash_map = cluster_hash_map

# Then in _store_activations loop:
cluster_hash = self.cluster_hash_map[cluster_idx]  # No .to_string() call!
```

---

### =� MEDIUM IMPACT (2-10x speedup)

#### 4. Pre-compute component index map per cluster
```python
# During BatchProcessingStorage.create():
self.cluster_component_indices: dict[int, dict[str, int | None]] = {
    cluster_idx: {
        comp["label"]: None  # Will be filled during first batch processing
        for comp in cluster_components[cluster_idx]
    }
    for cluster_idx in unique_cluster_indices
}

# During first batch, populate:
if self.cluster_component_indices[cluster_idx][comp_label] is None:
    self.cluster_component_indices[cluster_idx][comp_label] = processed.get_label_index(comp_label)

# Then use cached value:
comp_idx = self.cluster_component_indices[cluster_idx][comp_label]
```

#### 5. Cache dictionary lookups - use local variables
```python
# Before inner loops:
cluster_acts_list = self.cluster_activations[cluster_hash]
cluster_hashes_list = self.cluster_text_hashes[cluster_hash]
cluster_tokens_list = self.cluster_tokens[cluster_hash]

# Then append to local references
for sample_idx in range(batch_size):
    cluster_acts_list.append(...)
    cluster_hashes_list.append(...)
    cluster_tokens_list.append(...)
```

#### 6. Filter empty clusters earlier
```python
# Before the cluster loop, find active clusters:
active_cluster_mask = np.abs(acts_3d_cpu).max(axis=(0,1)) > 0
active_cluster_indices = [
    (col_idx, cluster_idx)
    for col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices)
    if active_cluster_mask[col_idx]
]

# Then only iterate over active clusters:
for col_idx, cluster_idx in active_cluster_indices:
    # ... process only clusters with non-zero activations
```

#### 7. Batch text_hash and tokens extraction
```python
# Extract once before loops:
batch_text_hashes = [sample.text_hash for sample in batch_text_samples]
batch_tokens = [sample.tokens for sample in batch_text_samples]

# Then extend once instead of append in loop:
self.cluster_text_hashes[hash].extend(batch_text_hashes)
self.cluster_tokens[hash].extend(batch_tokens)
```

---

### =� LOW IMPACT (10-50% speedup)

8. **Use numpy operations** instead of Python loops where possible

9. **Pre-allocate lists** with known sizes to avoid reallocations

10. **Move invariant computations** out of loops (e.g., `self.cluster_components[cluster_idx]`)

---

## Recommended Implementation Order

1. **Start with #1 (GPU�CPU transfers)** - This alone could give 10-50x speedup
2. **Then #3 (pre-compute cluster_hash_map)** - Easy win, ~2x speedup
3. **Then #2 (vectorize with extend)** - Good speedup, cleaner code
4. **Then #5 (cache dict lookups)** - Easy, small win
5. **Then #6 (filter empty clusters)** - Avoids wasted iterations
6. **Then #7 (batch text extraction)** - Small cleanup

Expected combined speedup: **20-100x** on the storing step!

---

## Testing Strategy

- Time the function before and after each optimization
- Verify outputs are identical (use hash of final data structures)
- Test with different batch sizes and cluster counts
- Profile with `cProfile` to find remaining bottlenecks
