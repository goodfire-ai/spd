# Harvest merge OOM (resolved)

## Problem

Two separate OOM sources in the merge pipeline, both caused by Python object overhead for reservoir data (40K components × 1K examples × 41-token windows):

1. **Merge step** (fixed earlier): Two `HarvesterState` objects as Python lists = 340 GB.
2. **Finalization step** (fixed now): `Harvester.from_state()` unpacked tensor reservoirs back into Python lists (~187 GB), plus `build_results()` accumulated all 40K `ComponentData` objects in a list (~166 GB more).

## Solution

### Phase 1: Tensor-packed merge (earlier fix)
Pack reservoir data as dense tensors in `HarvesterState` (~26 GB vs ~170 GB as Python lists). Vectorized Efraimidis-Spirakis merge via gather+where.

### Phase 2: Streaming finalization (current fix)
- `from_state()` keeps tensor reservoirs directly (never unpacks to Python lists)
- `build_results()` is a generator (yields one `ComponentData` at a time)
- `save_components_iter()` streams components to SQLite without accumulating

## Memory profile

| Stage | Before | After |
|-------|--------|-------|
| Merge (2 states) | 340 GB (OOM) | ~96 GB |
| Finalization | ~235 GB (OOM at 209 GB) | ~61 GB |
