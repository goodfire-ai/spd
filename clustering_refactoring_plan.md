# Clustering Module Refactoring Plan

## Current Architecture Issues

The clustering module was implemented by an intern and follows a script-based approach rather than a modular library design. Key issues:

1. **Tight Coupling**: Main orchestrator handles too many responsibilities
2. **Complex Process Communication**: Uses file descriptors and JSON parsing
3. **Mixed Concerns**: Visualization, computation, and orchestration intermingled
4. **Hard-coded Dependencies**: Magic numbers and paths scattered throughout
5. **Limited Modularity**: Difficult to use components independently

## Refactoring Steps

### Step 1: Replace File Descriptor Communication

**Problem**: Complex FD-based inter-process communication in `scripts/main.py`

**Solution**: Replace with exit codes + structured file outputs

**Current approach:**
```python
proc, json_r = launch_child_with_json_fd(cmd)
result = _read_json_result(json_r, dataset_path)
```

**Proposed approach:**
```python
result_file = temp_dir / f"{dataset_path.stem}_result.json"
cmd.extend(["--result-file", str(result_file)])

proc = subprocess.run(cmd, capture_output=False)
if proc.returncode != 0:
    raise RuntimeError(f"Clustering failed: {dataset_path.stem}")

result = json.loads(result_file.read_text())
```

**Files to modify:**
- `spd/clustering/scripts/main.py` - Remove `launch_child_with_json_fd()`, `_read_json_result()`
- `spd/clustering/scripts/s2_run_clustering.py` - Replace `emit_result()` with file-based output

**Benefits:**
- Eliminates ~50 lines of complex FD management
- Platform-independent communication
- Easier debugging (can inspect result files)
- Cleaner error handling

### Step 2: Simplified Interface Design

**Problem**: Current interfaces mix computation, I/O, and orchestration, with too many optional parameters

**Proposed 3-Layer Architecture:**

#### Layer 1: Pure Computation (no I/O, no side effects)
```python
# Core clustering algorithm - simplified
def merge_iteration(
    activations: ProcessedActivations,
    config: MergeConfig,
) -> MergeHistory

# Cost computation
def compute_merge_costs(
    coact: Tensor,
    merges: GroupMerge,
    alpha: float,
) -> Tensor

# Ensemble normalization - works on objects
def normalize_ensemble(
    ensemble: MergeHistoryEnsemble
) -> NormalizedHistories

# Distance computation
def compute_distances(
    normalized: NormalizedHistories,
    method: DistancesMethod = "perm_invariant_hamming"
) -> DistancesArray
```

#### Layer 2: Data Processing (I/O + transformation)
```python
# Component extraction and processing
def extract_and_process_activations(
    model: ComponentModel,
    batch: Tensor,
    config: MergeConfig,
) -> ProcessedActivations

# History loading/saving
def load_merge_histories(paths: list[Path]) -> MergeHistoryEnsemble
def save_merge_history(history: MergeHistory, path: Path) -> None

# Batch processing
def process_data_batch(
    config: MergeRunConfig,
    batch_path: Path,
) -> MergeHistory
```

#### Layer 3: Orchestration (coordination, parallel execution, file management)
```python
# Main pipeline - thin orchestrator
def cluster_analysis_pipeline(
    config: MergeRunConfig
) -> ClusteringResults

# Batch coordination with proper parallelism
class BatchProcessor:
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers

    def process_all_batches(
        self,
        batches: list[Path],
        config: MergeRunConfig
    ) -> list[MergeHistory]:
        # Use multiprocessing.Pool instead of subprocess + FD communication
        with multiprocessing.Pool(self.n_workers) as pool:
            worker_args = [(batch, config) for batch in batches]
            histories = pool.starmap(self._process_single_batch, worker_args)
        return histories

    def _process_single_batch(self, batch_path: Path, config: MergeRunConfig) -> MergeHistory:
        # This runs in worker process - clean data transformation
        return process_data_batch(config, batch_path)  # Layer 2 function
```

**Key Changes:**
1. **Remove callback complexity** - plotting/logging handled externally
2. **Consistent data types** - functions work on objects, not file paths mixed with objects
3. **Single responsibility** - each function does one thing well
4. **Composable** - pure functions can be easily tested and combined

### Step 3: Extract Value Objects

**Problem**: Data frequently travels together but isn't grouped

**Proposed Data Classes:**
```python
@dataclass
class ProcessedActivations:
    activations: Tensor
    labels: list[str]
    metadata: dict[str, Any]

@dataclass
class NormalizedHistories:
    merges_array: MergesArray
    component_labels: list[str]
    metadata: dict[str, Any]

@dataclass
class ClusteringResults:
    histories: list[MergeHistory]
    normalized: NormalizedHistories
    distances: DistancesArray
    config: MergeRunConfig
```

### Step 4: Modern Parallelism Strategy

**Problem**: Current subprocess + FD approach is complex and fragile

**Solution**: Use Python's `multiprocessing.Pool` for clean parallel execution

**Current approach (complex):**
```python
# 100+ lines of subprocess management, FD passing, JSON serialization
proc, json_r = launch_child_with_json_fd(cmd)
result = _read_json_result(json_r, dataset_path)
```

**New approach (simple):**
```python
from multiprocessing import Pool

class BatchProcessor:
    def __init__(self, n_workers: int = 4, devices: list[str] | None = None):
        self.n_workers = n_workers
        self.devices = devices or ["cuda:0"]

    def process_all_batches(
        self,
        batches: list[Path],
        config: MergeRunConfig
    ) -> list[MergeHistory]:
        # Create worker arguments with device assignment
        worker_args = [
            (batch, config, self.devices[i % len(self.devices)])
            for i, batch in enumerate(batches)
        ]

        with Pool(self.n_workers) as pool:
            histories = pool.starmap(self._process_single_batch, worker_args)

        return histories

    @staticmethod
    def _process_single_batch(
        batch_path: Path,
        config: MergeRunConfig,
        device: str
    ) -> MergeHistory:
        """Runs in worker process - pure computation, no callbacks"""
        # Load model and data
        model = ComponentModel.from_pretrained(config.model_path).to(device)
        batch_data = torch.load(batch_path)

        # Extract and process activations (Layer 2)
        activations = extract_and_process_activations(model, batch_data, config)

        # Run clustering (Layer 1 - pure computation)
        history = merge_iteration(activations, config)

        return history  # Automatically serialized by multiprocessing
```

**Benefits:**
- **~90% code reduction** for parallel execution
- **Native Python** - no shell commands or FD management
- **Automatic serialization** - Python handles MergeHistory objects
- **Clean error propagation** - exceptions bubble up properly
- **Easy debugging** - can run single-threaded with `n_workers=1`
- **GPU isolation** - each process gets separate CUDA context

## Target Architecture

### Core Principles
1. **Preserve tensor math exactly** - don't touch the core algorithms
2. **Layer separation** - pure computation → I/O → orchestration
3. **Clean interfaces** - functions do one thing well
4. **Modern Python** - use multiprocessing, dataclasses, type hints
5. **Testable** - pure functions can be tested in isolation

### Step 4: Replace Subprocess Communication with multiprocessing.Pool

**Problem**: Current complex subprocess + FD communication system

**Current approach (100+ lines):**
- `launch_child_with_json_fd()` - complex FD setup
- `distribute_clustering()` - manual process management
- `_read_json_result()` - FD parsing
- Error-prone cross-platform FD handling

**New approach (10 lines):**
```python
with multiprocessing.Pool(n_workers) as pool:
    worker_args = [(batch, config) for batch in batches]
    histories = pool.starmap(process_single_batch, worker_args)
```

**Benefits:**
- **Native Python** - no shell commands or FD management
- **Automatic serialization** - Python handles data passing
- **Better error handling** - exceptions propagate properly
- **Still bypasses GIL** - each worker is separate Python interpreter
- **GPU isolation** - each process has separate CUDA context
- **Simpler debugging** - can run single-threaded easily

## Implementation Strategy

**⚠️ CRITICAL: Preserve Core Math**
- Do NOT modify functions in `spd/clustering/math/`
- Do NOT modify core tensor operations in `merge.py`, `compute_costs.py`
- These contain complex mathematical algorithms we don't fully understand
- Only refactor the **orchestration and I/O layers** around the math

**Implementation Order:**
1. **Step 1**: Replace FD communication (low risk)
2. **Step 2**: Extract pure computation interfaces (medium risk - but only interface changes)
3. **Step 3**: Create value objects (low risk)
4. **Step 4**: Replace subprocess with multiprocessing (low risk)

**Safety Principles:**
- Keep all existing math functions unchanged
- Preserve existing test behavior exactly
- Create new interfaces that **wrap** existing functions, don't modify them
- Extensive testing at each step

## Implementation Status

### ✅ Completed Refactoring

We successfully implemented a clean 3-layer architecture:

**Files Created:**
- `spd/clustering/core.py` - Pure computation layer
- `spd/clustering/data_processing.py` - I/O and data transformation
- `spd/clustering/orchestration.py` - Parallel execution with multiprocessing
- `spd/clustering/main_refactored.py` - Backward-compatible CLI wrapper
- `spd/clustering/test_refactoring.py` - Validation tests

### Key Achievements

1. **90% code reduction** in parallel execution (100+ lines → 10 lines)
2. **Clean separation** of computation, I/O, and orchestration
3. **Preserved exact tensor math** - core algorithms untouched
4. **Modern Python** - multiprocessing.Pool instead of subprocess+FDs
5. **Simplified interfaces** - `merge_iteration()` reduced from 8 to 3 parameters
6. **Backward compatible** - can be drop-in replacement

### Results

**Before (scripts/main.py):**
- 370 lines of complex orchestration
- 100+ lines for subprocess/FD management
- Mixed concerns (computation + I/O + parallelism)
- Hard to test and debug

**After (orchestration.py):**
- ~150 lines total for entire orchestration
- ~10 lines for parallel execution
- Clean layer separation
- Easy to test each layer independently

The refactored code produces functionally identical results while being much cleaner and more maintainable.

## Implementation Notes

- Keep backward compatibility where possible
- Maintain existing CLI interfaces during transition
- Add comprehensive tests for new components
- **DO NOT TOUCH THE MATH** - only refactor around it
- Preserve all existing functionality while improving structure