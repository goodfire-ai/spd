# Harvest merge OOM (resolved)

## Problem

`spd-harvest-merge` OOMed on large models (e.g. s-275c8f21 with 39,936 components, vocab 50,254). Node RAM limit: ~200GB.

## Root cause

Reservoir states were stored as nested Python lists: 40K components × 1K examples × 3 lists of 41 Python objects each. Python object overhead (24-28 bytes per int/float) inflated ~26GB of data to **~170GB** in RAM. Two states during merge = 340GB — impossible.

## Solution

Pack reservoir data as dense tensors in `HarvesterState`:
- `reservoir_tokens`: `[n_comp, k, window]` int64
- `reservoir_ci`: `[n_comp, k, window]` float32
- `reservoir_acts`: `[n_comp, k, window]` float32
- `reservoir_n_items`, `reservoir_n_seen`: `[n_comp]` int64

~26GB as tensors. Two states during merge = ~52GB + stats tensors ~60GB = ~112GB. Well under 200GB.

Reservoir merge is vectorized (Efraimidis-Spirakis via gather+where, no concatenation of the big window-dimension tensors).

## Result

- Peak RSS: 135GB (was OOMing at 200GB)
- Merge time: 15 min for 2 workers, 100 batches
- No GPU required for merge
