# Harvest merge OOM (resolved)

## Problem

Python object overhead for reservoir data (40K components × 1K examples × 41-token windows) caused OOM at multiple points in the pipeline.

## Solution

Store reservoirs as dense tensors throughout — during harvesting on GPU, during merge on CPU. No Python list representation ever exists.

- `Harvester` does reservoir sampling (Algorithm R) directly on `[n_comp, k, window]` tensors
- `get_state()` / `from_state()` are trivial `.cpu()` / `.to(device)`
- `build_results()` yields one component at a time, reads tensors directly
- `save_components_iter()` streams to SQLite

## Memory

| Stage | Python lists (old) | Tensors (current) |
|-------|-------------------|-------------------|
| Worker GPU | ~187 GB CPU | ~58 GB GPU |
| Merge (2 states) | 340 GB (OOM) | ~96 GB |
| Finalization | ~235 GB (OOM) | ~61 GB |
