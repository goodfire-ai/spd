# Dashboard Data Generation Pipeline: Performance Optimization Report

## Executive Summary

The dashboard data generation pipeline processes neural network component activations through a batch processing system. The critical bottleneck is in the `_store_activations()` method in `batch_storage.py`, which stores cluster-level and component-level activation data.

**Current Performance Issues:**
- Triple nested loop structure: O(n_clusters × batch_size × n_components_per_cluster)
- Example: 100 clusters × 32 samples × 5 components = **16,000 iterations per batch**
- **Potential speedup: 20-100x** with proposed optimizations

**Key Optimizations Already Applied:**
- ✅ GPU→CPU transfers moved outside loops (lines 238-251)
- ✅ Pre-computed cluster hash map in `create()` method (line 124)

**Remaining Opportunities:**
- Vectorize inner loops using batch operations
- Pre-compute component label indices
- Use list `.extend()` instead of repeated `.append()`
- Cache dictionary lookups in local variables
- Filter empty clusters earlier

---

## System Architecture

### Data Flow Overview

```
Input Batch (tokens)
    ↓
[ComponentModel.forward()] - Get component activations
    ↓
[process_activations()] - Filter dead components, concatenate
    ↓
[compute_all_cluster_activations()] - Compute cluster-level activations
    ↓
[_store_activations()] ← **BOTTLENECK**
    ↓
DashboardData (final output)
```

### Key Data Structures

#### 1. ProcessedActivations
```python
@dataclass(frozen=True)
class ProcessedActivations:
    activations: Tensor  # Shape: [batch_size * seq_len, n_components]
    labels: ComponentLabels  # List of "module:index" strings
    label_index: dict[str, int | None]  # Cached mapping from label → index

    def get_label_index(self, label: str) -> int | None:
        """Get component index in activations array, or None if dead"""
        return self.label_index[label]
```

#### 2. ClusterActivations
```python
@dataclass
class ClusterActivations:
    activations: Tensor  # Shape: [n_samples, n_clusters]
    cluster_indices: list[int]  # Maps column → cluster index
```

#### 3. BatchProcessingStorage
```python
@dataclass
class BatchProcessingStorage:
    # Storage dictionaries (keys are cluster hashes)
    cluster_activations: dict[ClusterIdHash, list[ndarray]]
    cluster_text_hashes: dict[ClusterIdHash, list[TextSampleHash]]
    cluster_tokens: dict[ClusterIdHash, list[list[str]]]
    component_activations: dict[ClusterIdHash, dict[str, list[ndarray]]]
    component_text_hashes: dict[ClusterIdHash, dict[str, list[TextSampleHash]]]

    # Pre-computed metadata
    cluster_id_map: dict[int, ClusterId]
    cluster_components: dict[int, list[dict]]  # cluster_idx → component info
    cluster_hash_map: dict[int, ClusterIdHash]  # Pre-computed hashes
```

#### 4. ClusterId and Hashing
```python
@dataclass(frozen=True)
class ClusterId:
    clustering_run: str
    iteration: int
    cluster_label: int

    def to_string(self) -> ClusterIdHash:
        """Returns 'run-iteration-label'"""
        return ClusterIdHash(f"{self.clustering_run}-{self.iteration}-{self.cluster_label}")
```

#### 5. TextSample
```python
@dataclass(frozen=True)
class TextSample:
    full_text: str
    tokens: list[str]

    @cached_property
    def text_hash(self) -> TextSampleHash:
        """SHA256 hash (first 8 chars) of full_text"""
        return TextSampleHash(hashlib.sha256(self.full_text.encode()).hexdigest()[:8])
```

---

## Current Bottleneck: `_store_activations()`

### Method Signature
```python
def _store_activations(
    self,
    cluster_acts: ClusterActivations,        # [n_samples, n_clusters]
    processed: ProcessedActivations,         # [n_samples, n_components]
    batch_text_samples: list[TextSample],    # len=batch_size
    batch_size: int,                         # e.g., 32
    seq_len: int,                            # e.g., 128
) -> None:
```

### Current Implementation Structure

```python
# OPTIMIZATION ALREADY APPLIED: Move GPU→CPU transfers outside loops
acts_3d_cpu = cluster_acts.activations.view(batch_size, seq_len, -1).cpu().numpy()
processed_acts_3d = processed.activations.cpu().numpy().reshape(batch_size, seq_len, -1)

# LOOP 1: Iterate over clusters
for cluster_col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices):
    cluster_acts_2d = acts_3d_cpu[:, :, cluster_col_idx]  # [batch_size, seq_len]

    # Skip empty clusters
    if np.abs(cluster_acts_2d).max() == 0:
        continue

    # OPTIMIZATION ALREADY APPLIED: Use pre-computed hash
    cluster_hash = self.cluster_hash_map[cluster_idx]

    # LOOP 2: Iterate over samples in batch
    for batch_sample_idx in range(batch_size):
        text_sample = batch_text_samples[batch_sample_idx]
        text_hash = text_sample.text_hash

        # Store cluster-level data
        activations_np = cluster_acts_2d[batch_sample_idx]  # [seq_len]
        self.cluster_activations[cluster_hash].append(activations_np)
        self.cluster_text_hashes[cluster_hash].append(text_hash)
        self.cluster_tokens[cluster_hash].append(text_sample.tokens)

        # LOOP 3: Iterate over components in cluster
        components_in_cluster = self.cluster_components[cluster_idx]
        for component_info in components_in_cluster:
            component_label = component_info["label"]

            # BOTTLENECK: Dictionary lookup in hot loop
            comp_idx = processed.get_label_index(component_label)

            if comp_idx is not None:
                # Pure numpy indexing (no GPU transfer)
                comp_acts_1d = processed_acts_3d[batch_sample_idx, :, comp_idx]

                # BOTTLENECK: Repeated dictionary lookups
                self.component_activations[cluster_hash][component_label].append(comp_acts_1d)
                self.component_text_hashes[cluster_hash][component_label].append(text_hash)
```

### Complexity Analysis

**Current Implementation:**
- Outer loop: `O(n_clusters)` - typically 50-200 clusters
- Middle loop: `O(batch_size)` - typically 32 samples
- Inner loop: `O(n_components_per_cluster)` - typically 5-10 components
- **Total: O(n_clusters × batch_size × n_components_per_cluster)**

**Example Workload:**
- 100 clusters × 32 samples × 5 components = **16,000 iterations**
- Each iteration:
  - 1 `get_label_index()` dict lookup
  - 2-3 dictionary appends (`cluster_activations`, `component_activations`, etc.)
  - Numpy array indexing operations

**Measured Bottlenecks:**
1. ❌ **Repeated `.append()` calls** - List appends can cause reallocations
2. ❌ **Repeated dictionary lookups in hot loops** - Same keys looked up thousands of times
3. ❌ **`get_label_index()` called inside hot loop** - Should be pre-computed per cluster
4. ❌ **Not vectorized** - Processing samples one at a time instead of batched

---

## Proposed Optimizations

### High Impact (10-100x speedup potential)

#### 1. Vectorize Storage - Use `.extend()` Instead of Loop Appends

**Problem:** Currently appending to lists individually inside nested loops
```python
for batch_sample_idx in range(batch_size):
    self.cluster_activations[cluster_hash].append(cluster_acts_2d[batch_sample_idx])
    self.cluster_text_hashes[cluster_hash].append(text_hash)
    self.cluster_tokens[cluster_hash].append(text_sample.tokens)
```

**Solution:** Batch the data and use single `.extend()` calls
```python
# Extract all batch data at cluster level ONCE
batch_cluster_acts = [cluster_acts_2d[i] for i in range(batch_size)]
batch_text_hashes = [batch_text_samples[i].text_hash for i in range(batch_size)]
batch_tokens = [batch_text_samples[i].tokens for i in range(batch_size)]

# Single extend calls (much faster than repeated appends)
self.cluster_activations[cluster_hash].extend(batch_cluster_acts)
self.cluster_text_hashes[cluster_hash].extend(batch_text_hashes)
self.cluster_tokens[cluster_hash].extend(batch_tokens)
```

**Expected Speedup:** 5-10x for this portion

---

#### 2. Pre-compute Component Index Map per Cluster

**Problem:** `processed.get_label_index(component_label)` called inside hot loop
```python
for component_info in components_in_cluster:
    component_label = component_info["label"]
    comp_idx = processed.get_label_index(component_label)  # Called batch_size times!
```

**Solution A: Pre-compute during first batch** (Lazy initialization)
```python
# Add to BatchProcessingStorage.__init__
self.cluster_component_indices: dict[int, dict[str, int | None]] = {}

# In _store_activations, before batch loop:
if cluster_idx not in self.cluster_component_indices:
    # First time seeing this cluster - build index map
    self.cluster_component_indices[cluster_idx] = {
        comp["label"]: processed.get_label_index(comp["label"])
        for comp in self.cluster_components[cluster_idx]
    }

comp_idx_map = self.cluster_component_indices[cluster_idx]

# Then in inner loop:
for component_info in components_in_cluster:
    comp_label = component_info["label"]
    comp_idx = comp_idx_map[comp_label]  # Fast dict lookup, no method call
```

**Solution B: Pre-compute during `create()`** (Eager initialization)
```python
# In BatchProcessingStorage.create():
cluster_component_indices: dict[int, dict[str, int | None]] = {}
for cluster_idx in unique_cluster_indices:
    cluster_component_indices[cluster_idx] = {
        comp["label"]: None  # Will be populated on first batch
        for comp in cluster_components[cluster_idx]
    }

# Then populate during first batch processing
# (Same as Solution A, but structure is pre-allocated)
```

**Recommendation:** Use Solution A (lazy initialization) since `processed` object changes between batches and contains different components. The index map must be built fresh for each batch, but can be reused within the batch.

**Expected Speedup:** 2-5x

---

#### 3. Cache Dictionary Lookups - Use Local Variables

**Problem:** Repeated dictionary lookups in inner loops
```python
for batch_sample_idx in range(batch_size):
    # These dictionary lookups happen batch_size times per cluster
    self.cluster_activations[cluster_hash].append(...)
    self.cluster_text_hashes[cluster_hash].append(...)
    self.cluster_tokens[cluster_hash].append(...)
```

**Solution:** Cache references before loops
```python
# Before the sample loop - cache all dictionary references
cluster_acts_list = self.cluster_activations[cluster_hash]
cluster_hashes_list = self.cluster_text_hashes[cluster_hash]
cluster_tokens_list = self.cluster_tokens[cluster_hash]

comp_acts_dict = self.component_activations[cluster_hash]
comp_hashes_dict = self.component_text_hashes[cluster_hash]

# Then use cached references (faster)
for batch_sample_idx in range(batch_size):
    cluster_acts_list.append(...)
    cluster_hashes_list.append(...)
    cluster_tokens_list.append(...)

    for component_info in components_in_cluster:
        comp_label = component_info["label"]
        comp_acts_dict[comp_label].append(...)
        comp_hashes_dict[comp_label].append(...)
```

**Note:** This optimization is most effective when combined with vectorization (#1), otherwise savings are minimal.

**Expected Speedup:** 1.2-2x when used alone, 1.1-1.2x when combined with vectorization

---

### Medium Impact (2-10x speedup)

#### 4. Filter Empty Clusters Earlier with Vectorized Operations

**Problem:** Currently checking each cluster individually inside loop
```python
for cluster_col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices):
    cluster_acts_2d = acts_3d_cpu[:, :, cluster_col_idx]
    if np.abs(cluster_acts_2d).max() == 0:
        continue
```

**Solution:** Vectorized filtering before the loop
```python
# Compute max activation for each cluster across all samples and positions
# Shape: [n_clusters]
cluster_max_acts = np.abs(acts_3d_cpu).max(axis=(0, 1))

# Get indices of active clusters (activation > 0)
active_mask = cluster_max_acts > 0
active_cluster_indices = [
    (col_idx, cluster_idx)
    for col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices)
    if active_mask[col_idx]
]

# Iterate only over active clusters
for col_idx, cluster_idx in active_cluster_indices:
    cluster_acts_2d = acts_3d_cpu[:, :, col_idx]
    cluster_hash = self.cluster_hash_map[cluster_idx]
    # ... rest of processing
```

**Expected Speedup:** 1.5-3x if many clusters are inactive (typical in sparse activation patterns)

---

#### 5. Batch Text Hash and Token Extraction

**Problem:** Extracting text data inside sample loop
```python
for batch_sample_idx in range(batch_size):
    text_sample = batch_text_samples[batch_sample_idx]
    text_hash = text_sample.text_hash
    # ... append text_hash and text_sample.tokens repeatedly
```

**Solution:** Extract once before loops
```python
# Before cluster loop - extract all batch metadata once
batch_text_hashes = [sample.text_hash for sample in batch_text_samples]
batch_tokens = [sample.tokens for sample in batch_text_samples]

# Then in cluster loop, extend in batch:
for cluster_col_idx, cluster_idx in enumerate(active_cluster_indices):
    # ... cluster processing ...

    # Batch extend (much faster than loop appends)
    self.cluster_text_hashes[cluster_hash].extend(batch_text_hashes)
    self.cluster_tokens[cluster_hash].extend(batch_tokens)
```

**Expected Speedup:** 1.5-2x

---

#### 6. Vectorize Component Activation Storage

**Problem:** Storing component activations one sample at a time
```python
for batch_sample_idx in range(batch_size):
    for component_info in components_in_cluster:
        comp_acts_1d = processed_acts_3d[batch_sample_idx, :, comp_idx]
        self.component_activations[cluster_hash][component_label].append(comp_acts_1d)
```

**Solution:** Vectorized extraction and batch storage
```python
# Build component index map once per cluster
comp_indices_in_cluster = []
comp_labels_in_cluster = []
for component_info in components_in_cluster:
    comp_label = component_info["label"]
    comp_idx = processed.get_label_index(comp_label)
    if comp_idx is not None:
        comp_indices_in_cluster.append(comp_idx)
        comp_labels_in_cluster.append(comp_label)

# Extract all component activations for all samples in batch at once
# Shape: [batch_size, seq_len, n_comps_in_cluster]
comp_acts_batch = processed_acts_3d[:, :, comp_indices_in_cluster]

# Store per component (still need to iterate over components)
for i, comp_label in enumerate(comp_labels_in_cluster):
    # Extract this component's activations for all samples: [batch_size, seq_len]
    comp_acts_all_samples = comp_acts_batch[:, :, i]

    # Convert to list of arrays and extend
    comp_acts_list = [comp_acts_all_samples[j] for j in range(batch_size)]
    self.component_activations[cluster_hash][comp_label].extend(comp_acts_list)
    self.component_text_hashes[cluster_hash][comp_label].extend(batch_text_hashes)
```

**Expected Speedup:** 3-5x for component storage

---

### Low Impact (10-50% speedup)

#### 7. Pre-allocate Lists with Known Sizes

**Problem:** Lists grow dynamically, may cause reallocations
```python
# Current: Lists initialized as empty
cluster_activations: dict[ClusterIdHash, list] = {hash: [] for hash in hashes}
```

**Solution:** Pre-allocate with estimated capacity (Python doesn't support this directly, but we can do it indirectly)
```python
# Python doesn't have direct pre-allocation, but we can reduce reallocations
# by initializing with dummy data if we know the size
# However, this is complex for our case since we don't know batch size ahead of time
# and clusters have variable activity.

# SKIP this optimization - not worth the complexity
```

---

#### 8. Move Invariant Computations Out of Loops

**Problem:** Computing the same value repeatedly
```python
for batch_sample_idx in range(batch_size):
    components_in_cluster = self.cluster_components[cluster_idx]  # Same every iteration
    for component_info in components_in_cluster:
        # ...
```

**Solution:** Compute once before loop
```python
components_in_cluster = self.cluster_components[cluster_idx]  # Move outside sample loop
for batch_sample_idx in range(batch_size):
    for component_info in components_in_cluster:
        # ...
```

**Expected Speedup:** 5-10% (small improvement, but easy)

---

## Optimized Implementation

### Complete Optimized `_store_activations()` Method

```python
def _store_activations(
    self,
    cluster_acts: ClusterActivations,
    processed: ProcessedActivations,
    batch_text_samples: list[TextSample],
    batch_size: int,
    seq_len: int,
) -> None:
    """Store cluster-level and component-level activations from batch.

    OPTIMIZED VERSION with vectorized operations and minimal redundant work.
    """
    # Store for coactivation computation
    self.all_cluster_activations.append(cluster_acts)

    # ========================================
    # OPTIMIZATION: GPU→CPU transfers outside loops (ALREADY APPLIED)
    # ========================================
    acts_3d = cluster_acts.activations.view(batch_size, seq_len, -1)
    acts_3d_cpu = acts_3d.cpu().numpy()  # [batch_size, seq_len, n_clusters]

    processed_acts_cpu = processed.activations.cpu().numpy()
    processed_acts_3d = processed_acts_cpu.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, n_components]

    # ========================================
    # OPTIMIZATION #5: Batch text extraction
    # ========================================
    batch_text_hashes = [sample.text_hash for sample in batch_text_samples]
    batch_tokens = [sample.tokens for sample in batch_text_samples]

    # ========================================
    # OPTIMIZATION #4: Filter empty clusters early
    # ========================================
    cluster_max_acts = np.abs(acts_3d_cpu).max(axis=(0, 1))  # [n_clusters]
    active_mask = cluster_max_acts > 0
    active_clusters = [
        (col_idx, cluster_idx)
        for col_idx, cluster_idx in enumerate(cluster_acts.cluster_indices)
        if active_mask[col_idx]
    ]

    # ========================================
    # OPTIMIZATION #2: Pre-compute component indices per cluster
    # ========================================
    # Build component index maps for all active clusters
    if not hasattr(self, '_component_idx_cache'):
        self._component_idx_cache = {}

    # ========================================
    # Main loop over ACTIVE clusters only
    # ========================================
    for col_idx, cluster_idx in tqdm(
        active_clusters,
        desc="  Storing cluster activations",
        leave=False,
    ):
        cluster_acts_2d = acts_3d_cpu[:, :, col_idx]  # [batch_size, seq_len]
        cluster_hash = self.cluster_hash_map[cluster_idx]  # Pre-computed hash

        # ========================================
        # OPTIMIZATION #1: Vectorize cluster-level storage
        # ========================================
        batch_cluster_acts = [cluster_acts_2d[i] for i in range(batch_size)]
        self.cluster_activations[cluster_hash].extend(batch_cluster_acts)
        self.cluster_text_hashes[cluster_hash].extend(batch_text_hashes)
        self.cluster_tokens[cluster_hash].extend(batch_tokens)

        # ========================================
        # OPTIMIZATION #6: Vectorize component storage
        # ========================================
        components_in_cluster = self.cluster_components[cluster_idx]  # Move outside loop

        # Build component index map for this cluster (cache across batches if possible)
        cache_key = (cluster_idx, id(processed))  # Use object id since processed changes
        if cache_key not in self._component_idx_cache:
            comp_idx_map = {
                comp["label"]: processed.get_label_index(comp["label"])
                for comp in components_in_cluster
            }
            self._component_idx_cache[cache_key] = comp_idx_map
        else:
            comp_idx_map = self._component_idx_cache[cache_key]

        # Filter to only alive components
        alive_comp_info = [
            (comp["label"], comp_idx_map[comp["label"]])
            for comp in components_in_cluster
            if comp_idx_map[comp["label"]] is not None
        ]

        if not alive_comp_info:
            continue

        # Extract labels and indices
        comp_labels, comp_indices = zip(*alive_comp_info)
        comp_indices_array = list(comp_indices)

        # Vectorized extraction: [batch_size, seq_len, n_comps_in_cluster]
        comp_acts_batch = processed_acts_3d[:, :, comp_indices_array]

        # ========================================
        # OPTIMIZATION #3: Cache dictionary lookups
        # ========================================
        comp_acts_storage = self.component_activations[cluster_hash]
        comp_hashes_storage = self.component_text_hashes[cluster_hash]

        # Store per component
        for i, comp_label in enumerate(comp_labels):
            # Extract all samples for this component: [batch_size, seq_len]
            comp_acts_all = comp_acts_batch[:, :, i]

            # Convert to list and extend
            comp_acts_list = [comp_acts_all[j] for j in range(batch_size)]
            comp_acts_storage[comp_label].extend(comp_acts_list)
            comp_hashes_storage[comp_label].extend(batch_text_hashes)
```

---

## Expected Performance Improvements

### Conservative Estimates

| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Baseline | 1x | 1x |
| + Batch text extraction (#5) | 1.5x | 1.5x |
| + Filter empty clusters (#4) | 2x | 3x |
| + Pre-compute component indices (#2) | 2x | 6x |
| + Cache dict lookups (#3) | 1.5x | 9x |
| + Vectorize storage (#1, #6) | 3x | **27x** |

### Optimistic Estimates

With ideal conditions (many inactive clusters, large batches, many components):
- Individual optimizations compound multiplicatively
- Expected speedup: **50-100x**

### Realistic Estimate

For typical workloads:
- **20-40x speedup** is achievable

---

## Implementation Plan

### Phase 1: Low-Risk, High-Impact (Do First)
1. ✅ **Move GPU→CPU transfers outside loops** - ALREADY DONE
2. ✅ **Pre-compute cluster hash map** - ALREADY DONE
3. **Batch text extraction (#5)** - Simple, safe, 1.5x speedup
4. **Filter empty clusters (#4)** - Simple, safe, 2x speedup
5. **Move invariant computations (#8)** - Trivial, 5-10% speedup

**Expected Combined: 3-4x speedup, minimal risk**

### Phase 2: Moderate Complexity, High Impact
6. **Pre-compute component indices (#2)** - Moderate complexity, 2-5x speedup
7. **Cache dictionary lookups (#3)** - Simple, works well with #6

**Expected Combined with Phase 1: 10-15x speedup**

### Phase 3: Higher Complexity, Highest Impact
8. **Vectorize cluster storage (#1)** - Refactor loops, 3-5x additional speedup
9. **Vectorize component storage (#6)** - Complex, but big win

**Expected Final: 20-40x total speedup**

---

## Testing Strategy

### 1. Correctness Testing
```python
def test_optimized_store_activations():
    # Run both old and new implementations
    storage_old = BatchProcessingStorage.create(cluster_id_map, cluster_components)
    storage_new = BatchProcessingStorage.create(cluster_id_map, cluster_components)

    # Process same batch
    storage_old._store_activations_old(cluster_acts, processed, batch_text_samples, batch_size, seq_len)
    storage_new._store_activations(cluster_acts, processed, batch_text_samples, batch_size, seq_len)

    # Compare outputs
    assert_storage_equal(storage_old, storage_new)

def assert_storage_equal(s1, s2):
    # Check all dictionaries have same keys
    assert s1.cluster_activations.keys() == s2.cluster_activations.keys()

    # Check values match
    for key in s1.cluster_activations:
        assert len(s1.cluster_activations[key]) == len(s2.cluster_activations[key])
        for arr1, arr2 in zip(s1.cluster_activations[key], s2.cluster_activations[key]):
            np.testing.assert_array_equal(arr1, arr2)

    # Similar checks for other dictionaries
    # ...
```

### 2. Performance Testing
```python
import time

def benchmark_store_activations(storage, cluster_acts, processed, batch_text_samples, batch_size, seq_len, n_runs=10):
    times = []
    for _ in range(n_runs):
        storage_copy = copy.deepcopy(storage)
        start = time.perf_counter()
        storage_copy._store_activations(cluster_acts, processed, batch_text_samples, batch_size, seq_len)
        times.append(time.perf_counter() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }

# Compare old vs new
old_stats = benchmark_store_activations(storage, ...)
new_stats = benchmark_store_activations(storage, ...)
speedup = old_stats['mean'] / new_stats['mean']
print(f"Speedup: {speedup:.2f}x")
```

### 3. Profiling
```python
import cProfile
import pstats

# Profile the function
profiler = cProfile.Profile()
profiler.enable()
storage._store_activations(cluster_acts, processed, batch_text_samples, batch_size, seq_len)
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 4. Validation Checklist
- [ ] All cluster activations match between old/new implementations
- [ ] All component activations match between old/new implementations
- [ ] Text hashes match
- [ ] Tokens match
- [ ] Performance improves by expected amount
- [ ] Memory usage doesn't increase significantly
- [ ] Works with edge cases:
  - [ ] Empty clusters
  - [ ] Single cluster
  - [ ] Large batch size (e.g., 128)
  - [ ] Small batch size (e.g., 1)
  - [ ] Dead components

---

## Additional Considerations

### Memory Usage
- Current approach: Lists grow dynamically via `.append()`
- Optimized approach: Still uses lists, but with `.extend()` for batch operations
- **Memory impact:** Negligible difference - same final size, slightly less fragmentation

### Numerical Stability
- All operations use the same numpy arrays and data types
- No floating-point arithmetic changes
- **Risk:** None

### Thread Safety
- Current code is not thread-safe (appending to shared lists)
- Optimized code maintains same threading model
- **Risk:** None (if already single-threaded)

### Code Maintainability
- Optimized code is more complex but better documented
- Clear separation of phases (batch extraction, filtering, storage)
- More explicit about what's happening (less magic)
- **Recommendation:** Add inline comments explaining each optimization

---

## Appendix: Key Code References

### Data Structure Dimensions

```python
# Input shapes:
cluster_acts.activations      # [n_samples, n_clusters]
                              # where n_samples = batch_size * seq_len

processed.activations         # [n_samples, n_components_total]

batch_text_samples            # list[TextSample], len = batch_size

# After reshaping:
acts_3d_cpu                   # [batch_size, seq_len, n_clusters]
processed_acts_3d             # [batch_size, seq_len, n_components_total]

# Per cluster:
cluster_acts_2d               # [batch_size, seq_len]

# Per component per cluster:
comp_acts_1d                  # [seq_len]
```

### Critical Method: `ProcessedActivations.get_label_index()`

```python
@cached_property
def label_index(self) -> dict[str, int | None]:
    """Mapping from component label to index in activations array.
    Returns None for dead components."""
    return {
        **{label: i for i, label in enumerate(self.labels)},
        **{label: None for label in self.dead_components_lst} if self.dead_components_lst else {},
    }

def get_label_index(self, label: str) -> int | None:
    """Get component index, or None if dead."""
    return self.label_index[label]  # Dict lookup on cached property
```

**Performance Note:** This is a cached property (computed once per `ProcessedActivations` object), so the dict is built only once. However, calling this method in a hot loop still incurs dict lookup overhead.

### Storage Dictionary Structure

```python
# Cluster-level storage:
cluster_activations[cluster_hash] = [
    np.ndarray([seq_len]),  # Sample 1
    np.ndarray([seq_len]),  # Sample 2
    # ...
]

# Component-level storage:
component_activations[cluster_hash][component_label] = [
    np.ndarray([seq_len]),  # Sample 1
    np.ndarray([seq_len]),  # Sample 2
    # ...
]
```

Each list grows as batches are processed. Final size depends on:
- Number of batches processed
- Batch size
- Cluster activity (how often each cluster activates)

---

## Summary

The `_store_activations()` bottleneck can be dramatically improved through:

1. **Vectorization** - Process batches of samples together
2. **Pre-computation** - Calculate indices and metadata once
3. **Caching** - Avoid redundant lookups
4. **Early filtering** - Skip unnecessary work

**Expected overall speedup: 20-40x** with all optimizations applied.

**Implementation priority:**
1. Start with simple, safe optimizations (Phase 1) - 3-4x speedup
2. Add moderate complexity optimizations (Phase 2) - 10-15x cumulative
3. Full vectorization (Phase 3) - 20-40x final speedup

**Risk level:** Low - All optimizations preserve exact numerical outputs, only change the order and batching of operations.
