# Harvest merge OOM

## Problem

`spd-harvest-merge` OOMs on large models (e.g. s-275c8f21 with 39,936 components, vocab 50,254).

Worker state files are ~44GB each. The dominant data per state:

| Data | Shape / Count | Size |
|------|---------------|------|
| `count_ij` | 39936 × 39936, float32 | 6.0 GB |
| `input_token_counts` | 39936 × 50254, int64 | 16.1 GB |
| `output_token_prob_mass` | 39936 × 50254, float32 | 8.0 GB |
| reservoir states | 39936 samplers × 1000 examples | ~40 GB (with Python object overhead) |

These sizes are independent of `n_batches` — fixed-size accumulators proportional to `n_components² + n_components × vocab_size`.

## Failed attempts

1. **128G RAM, CPU-only**: OOM loading second worker state (~88GB tensors alone)
2. **200G RAM, CPU-only**: Same
3. **200G RAM, mmap for workers 1-7**: OOM — single `+=` pages in entire mmaped tensor
4. **200G RAM, mmap + chunked iadd**: OOM — reservoir states (~40GB Python objects each) still need full materialization, two sets during merge = ~80GB + tensors
5. **200G RAM, two-pass (reservoirs then tensors), all mmap**: Barely fits at 190GB RSS — one more gc hiccup and it dies

## Solution

Two-pass merge with **1 GPU** — tensors on VRAM, reservoirs on CPU RAM:

- **Pass 1 (reservoirs)**: mmap each worker (tensors stay on disk), extract only `reservoir_states`, merge pairwise, discard. Peak: ~80GB RAM (two sets of Python reservoir objects).
- **Pass 2 (tensors)**: clone worker 0's tensors onto GPU (~30GB VRAM), stream remaining workers via mmap + chunked `iadd_chunked`. Peak: ~30GB VRAM + negligible CPU RSS.

This cleanly separates the two memory-hungry data types onto different memory pools (CPU RAM vs GPU VRAM), keeping each well within limits.
