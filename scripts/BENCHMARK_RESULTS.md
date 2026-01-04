# torch.compile Benchmark Results for SPD

Benchmarks comparing eager mode vs `torch.compile()` for SPD's masked forward/backward passes.

## Summary

- **Isolated LinearComponents**: No benefit from torch.compile (0-20% slower)
- **Full ComponentModel**: 30-35% speedup at larger batch sizes

## Full Model Results (SS Llama Simple)

Model: 4-layer Llama with 28 decomposed modules, C=1200 components each.

| Batch | Seq Len | Compile Mode | Eager | Compiled | Speedup |
|-------|---------|--------------|-------|----------|---------|
| 16 | 128 | reduce-overhead | 10.56ms | 9.64ms | **1.09x (9.5%)** |
| 32 | 256 | reduce-overhead | 12.61ms | 11.20ms | **1.13x (12.6%)** |
| 64 | 256 | reduce-overhead | 20.70ms | 15.96ms | **1.30x (30%)** |
| 128 | 256 | reduce-overhead | 36.19ms | 28.34ms | **1.28x (28%)** |
| 64 | 256 | max-autotune | 20.70ms | 15.38ms | **1.35x (35%)** |

## Isolated LinearComponents Results

Testing just the core operation: `out = (x @ V * mask) @ U`

| Test | Dimensions | Compile Mode | Speedup |
|------|------------|--------------|---------|
| LinearComponents | d=512, C=512 | reduce-overhead | 1.00x (no benefit) |
| LinearComponents | d=2048, C=2048 | reduce-overhead | 1.03x (2.6%) |
| Pure function | d=512, C=512 | reduce-overhead | 0.85x (15% slower) |
| Pure function | d=4096, C=4096 | reduce-overhead | 0.96x (4% slower) |
| FP16 | d=512, C=512 | reduce-overhead | 0.78x (22% slower) |

## Why the Difference?

### Full model benefits because:
- Many operations can be fused (28 masked layers + attention + activations + norms)
- More compute per kernel launch amortizes dispatch overhead
- torch.compile can optimize the entire forward/backward graph

### Isolated LinearComponents don't benefit because:
- The operation `(x @ V * mask) @ U` is just 2 matmuls + 1 multiply
- cuBLAS is already highly optimized for matmuls
- No fusion opportunities between matmuls
- Dispatch overhead exceeds any micro-optimization gains

## Recommendations

1. **Use torch.compile for full SPD training** with `mode="reduce-overhead"` or `mode="max-autotune"`
2. **Use larger batch sizes** (64+) to maximize compile benefits
3. **Don't bother compiling isolated components** - eager is faster or equivalent
4. **Budget for warmup time** - first few steps are slow due to compilation (~24s for reduce-overhead, ~70s for max-autotune)

## Reproduce

```bash
# Full model benchmark
python scripts/benchmark_full_model.py --batch_size 64 --seq_len 256

# Isolated LinearComponents benchmark
python scripts/benchmark_linear_components.py components --C 512
python scripts/benchmark_linear_components.py pure --d_in 512 --d_out 512 --C 512
```

## Hardware

- GPU: NVIDIA (CUDA)
- PyTorch with torch.compile (inductor backend)
