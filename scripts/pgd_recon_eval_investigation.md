# PGDReconLoss Standalone Eval vs Training Eval Discrepancy

## Objective

Evaluate the model from `wandb:goodfire/spd/runs/s-b9582efc` with `PGDReconLoss` at
varying `n_steps` (1, 5, 20, 100) and compare to the training-time eval metric.

## Run Configuration

The run uses:
- **Model**: `LlamaSimpleMLP` (2L)
- **output_loss_type**: `kl`
- **use_delta_component**: `True`
- **batch_size**: 64 (training), **eval_batch_size**: 64
- **DDP**: 8 GPUs, so per-rank batch size = 8
- **Eval PGD config**: `init=random, step_size=0.1, n_steps=20, mask_scope=shared_across_batch`
- **n_eval_steps**: 5

## Training-Time PGDReconLoss Values (from WandB)

At the end of training (steps 390k-400k), `eval/loss/PGDReconLoss` hovers around **2.5-4.3**:

```
step=392000  PGDReconLoss=3.37
step=393000  PGDReconLoss=4.25
step=394000  PGDReconLoss=2.54
step=395000  PGDReconLoss=3.42
step=396000  PGDReconLoss=2.91
step=397000  PGDReconLoss=2.76
step=398000  PGDReconLoss=3.20
step=399000  PGDReconLoss=2.36
step=400000  PGDReconLoss=3.00
```

## Standalone Eval Results

### Attempt 1: `unique_per_datapoint` scope, batch_size=8 (WRONG scope)

| n_steps | PGDReconLoss |
|---------|-------------|
| 1       | 3.10        |
| 5       | 28.20       |
| 20      | 48.51       |
| 100     | 57.52       |

This used the wrong `mask_scope`. The run's eval config uses `shared_across_batch`.

### Attempt 2: `shared_across_batch` scope, batch_size=8

| n_steps | PGDReconLoss |
|---------|-------------|
| 1       | 0.33        |
| 5       | 2.32        |
| 20      | 12.64       |
| 100     | 20.54       |

Correct scope, but **n_steps=20 gives 12.64** vs the training-time value of ~3. That's a ~4x discrepancy.

### Attempt 3: `shared_across_batch` scope, batch_size=64 (to match global batch)

OOM on a single H200 GPU — the KL divergence computation with batch_size=64 requires
more memory than available.

## Root Cause of the Discrepancy

The discrepancy stems from how `shared_across_batch` PGD works in DDP vs standalone.

### During training eval (8 GPUs, DDP)

In `spd/metrics/pgd_utils.py:64-77`, each PGD step:

1. Each rank runs a forward pass on its **local batch of 8 examples** with the shared
   adversarial sources (shape `[1, ..., mask_c]`)
2. Each rank computes the gradient of the mean loss wrt the shared sources
3. **Gradients are averaged across all 8 ranks** via `all_reduce(g, op=ReduceOp.AVG)` (line 71)
4. Sources are updated using the sign of the **averaged** gradient

This means the PGD attack is effectively optimizing against **64 examples simultaneously**
(8 examples/rank x 8 ranks). The gradient averaging means different examples' gradient
contributions can cancel each other out, weakening the attack.

### During standalone eval (1 GPU, no DDP)

`all_reduce` is a no-op when `is_distributed()` is False (`spd/utils/distributed_utils.py:185-187`).
So PGD optimizes the shared sources against only **8 examples**. Finding a single adversarial
mask that hurts 8 examples is much easier than finding one that hurts 64 examples, so the
attack is stronger and the resulting loss is higher.

### Why this matters

The `shared_across_batch` scope constrains PGD to find a **single set of adversarial sources**
that must work across all examples in the effective batch. A larger effective batch makes the
attack weaker because:
- Different examples may have conflicting gradient directions for the shared sources
- The sign of the averaged gradient may differ from any individual example's gradient
- The adversary has less freedom to specialize the attack to any particular example

### Attempt 4: Simulated 8-rank DDP on single GPU (CONFIRMED MATCH)

Manually replicated the DDP behavior: split each global batch of 64 into 8 sub-batches
of 8, compute PGD gradients on each sub-batch independently, average the gradients, then
take sign step. This exactly mirrors what `all_reduce(AVG)` does across ranks.

| n_steps | PGDReconLoss |
|---------|-------------|
| 1       | 0.27        |
| 5       | 0.57        |
| 20      | **3.21**    |
| 100     | 16.48       |

**n_steps=20 gives 3.21**, which falls squarely within the training-time range of 2.5-4.3.

## Summary of All Results

| n_steps | unique_per_datapoint (bs=8) | shared_across_batch (bs=8) | Simulated 8-rank DDP | Training eval (WandB) |
|---------|-----------------------------|---------------------------|---------------------|-----------------------|
| 1       | 3.10                        | 0.33                      | 0.27                | —                     |
| 5       | 28.20                       | 2.32                      | 0.57                | —                     |
| 20      | 48.51                       | 12.64                     | **3.21**            | **~2.5-4.3**          |
| 100     | 57.52                       | 20.54                     | 16.48               | —                     |

## Root Cause of the Discrepancy

The discrepancy stems from how `shared_across_batch` PGD works in DDP vs standalone.

### During training eval (8 GPUs, DDP)

In `spd/metrics/pgd_utils.py:64-77`, each PGD step:

1. Each rank runs a forward pass on its **local batch of 8 examples** with the shared
   adversarial sources (shape `[1, ..., mask_c]`)
2. Each rank computes the gradient of the mean loss wrt the shared sources
3. **Gradients are averaged across all 8 ranks** via `all_reduce(g, op=ReduceOp.AVG)` (line 71)
4. Sources are updated using the sign of the **averaged** gradient

This means the PGD attack is effectively optimizing against **64 examples simultaneously**
(8 examples/rank x 8 ranks). The gradient averaging means different examples' gradient
contributions can cancel each other out, weakening the attack.

### During standalone eval (1 GPU, no DDP)

`all_reduce` is a no-op when `is_distributed()` is False (`spd/utils/distributed_utils.py:185-187`).
So PGD optimizes the shared sources against only **8 examples**. Finding a single adversarial
mask that hurts 8 examples is much easier than finding one that hurts 64 examples, so the
attack is stronger and the resulting loss is higher.

### Why this matters

The `shared_across_batch` scope constrains PGD to find a **single set of adversarial sources**
that must work across all examples in the effective batch. A larger effective batch makes the
attack weaker because:
- Different examples may have conflicting gradient directions for the shared sources
- The sign of the averaged gradient may differ from any individual example's gradient
- The adversary has less freedom to specialize the attack to any particular example

## Conclusion

The standalone eval with batch_size=8 on 1 GPU is **not equivalent** to the training eval
with batch_size=8/rank on 8 GPUs for the `shared_across_batch` PGD metric. The training
eval effectively computes PGD gradients over 64 examples (via `all_reduce(AVG)` across
8 ranks), making the adversarial attack weaker and producing lower PGDReconLoss values.

Simulating the DDP gradient averaging on a single GPU (Attempt 4) confirmed the match:
**3.21 vs ~2.5-4.3** at n_steps=20.

### Implications

The `shared_across_batch` PGD metric is **not comparable across different DDP configurations**.
Running the same model with 1 GPU vs 8 GPUs will give different PGDReconLoss values because
the effective batch size for gradient averaging changes. This is a property of how
`all_reduce(AVG)` interacts with the sign-based PGD update, not a bug per se, but something
to be aware of when comparing eval results across setups.
