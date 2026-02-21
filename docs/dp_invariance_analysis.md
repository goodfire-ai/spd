# Data Parallelism Invariance Analysis: `pile_llama_simple_mlp-4L`

Analysis of whether `--dp 8` vs `--dp 16` produces significantly different results for the `pile_llama_simple_mlp-4L` config.

**Conclusion: Results should be nearly identical.** All losses are properly mean-reduced, DDP gradient averaging preserves global-batch-mean gradients, and rank-local operations (PPGD sources) are independent of the number of co-located sources per rank.

## Setup

| Setting | dp=8 | dp=16 |
|---------|------|-------|
| Global batch size | 64 | 64 |
| Per-rank microbatch | 8 | 4 |
| Eval batch per rank | 16 | 8 |
| Total PPGD sources | 8 ranks x 8 = 64 | 16 ranks x 4 = 64 |

The global batch size is fixed by config (`batch_size: 64`). Per-rank microbatch = `batch_size // world_size` (`lm_decomposition.py:97`).

## Training Losses

### PersistentPGDReconLoss (coeff=0.5, `PerBatchPerPositionScope`)

The config uses `per_batch_per_position` scope, which means:

- **Source shape**: `[B, S, C+1]` where B = per-rank microbatch (`persistent_pgd.py:158-159`)
- **No cross-rank gradient sync**: `_skip_all_reduce = True` for this scope (`persistent_pgd.py:137`)
- **Per-element gradients**: Each source at position `i` gets gradient only from data point `i` — no cross-batch-element interaction
- **Mean-reduced loss**: `sum_loss / n_examples` (`persistent_pgd.py:249`)

Since each source operates on exactly one (batch_element, position) pair independently, the number of sources co-located on a rank doesn't affect optimization. Total sources across all ranks is 64 in both cases.

The CI model parameters receive gradients through DDP's auto-averaging during `total_loss.backward()` (`run_spd.py:317`). Since the global batch is 64 in both cases, the effective CI gradient is the mean over 64 examples.

**DP-invariant.**

### StochasticReconSubsetLoss (coeff=0.5)

Standard mean-reduced loss. DDP auto-averages CI model gradients. Global batch is 64 in both cases, so the effective gradient is the mean over 64 examples.

**DP-invariant.**

### ImportanceMinimalityLoss (coeff=0.0004)

This loss explicitly compensates for data parallelism. During training (`importance_minimality_loss.py:138-144`):

```python
world_size = dist_state.world_size if dist_state is not None else 1
```

Used in the loss formula (`importance_minimality_loss.py:110-112`):

```python
per_component_mean + beta * per_component_mean * log2(1 + layer_sums * world_size)
```

- dp=8: `layer_sums` over 8 examples × 8 = estimates global sum over 64
- dp=16: `layer_sums` over 4 examples × 16 = estimates global sum over 64

`per_component_mean = layer_sums / n_examples` is the local mean, an unbiased estimator of the population mean regardless of local batch size.

During eval, sums are all-reduced first, then `world_size=1` is passed (`importance_minimality_loss.py:224`).

**DP-invariant in expectation.** Slightly higher variance with dp=16 (fewer local samples before multiplying up).

### FaithfulnessLoss (coeff=10M)

Mean-reduced during training. During eval, uses `all_reduce(SUM)` for both numerator and denominator to compute exact global average (`faithfulness_loss.py:56-57`).

**DP-invariant.**

## Eval Metrics

All eval metrics follow the same pattern:

```python
sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
return sum_loss / n_examples
```

This applies to: `PGDReconLoss`, `CEandKLLosses`, `StochasticHiddenActsReconLoss`, `UnmaskedReconLoss`, etc.

Global averages are exact regardless of how examples are distributed across ranks.

**DP-invariant.**

## Train Logging

Per-rank loss values (`loss.item()`) are averaged across ranks at log time (`run_spd.py:328`):

```python
avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
```

Since each rank computes the mean loss over its local microbatch and these means are averaged (mean-of-means with equal-sized microbatches = global mean), logged values are consistent.

**DP-invariant.**

## Sources of Minor Differences

1. **Gradient noise**: Smaller per-rank microbatch (4 vs 8) produces slightly noisier per-rank loss/gradient estimates before DDP averaging. Same mean, higher variance.
2. **Data ordering**: Different sharding patterns (`dataset.shard()` in `create_data_loader`) produce different batch compositions at each step.
3. **PPGD source initialization**: `broadcast_tensor` broadcasts from rank 0, but source tensors have different shapes (`[4, 512, C]` vs `[8, 512, C]`), so the random init differs.

None of these should produce systematically different training trajectories.
