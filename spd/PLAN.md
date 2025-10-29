# Plan: PGD Global Reconstruction Metrics

## Overview
Add new PGD global reconstruction loss metrics that accumulate gradients over multiple batches before taking PGD steps, unlike the existing single-batch PGD metrics.

## Changes Required

### 1. Config Classes (`spd/configs.py`)

Add three new config classes for the global PGD metrics:

```python
class PGDGlobalReconLossConfig(BaseConfig):
    classname: Literal["PGDGlobalReconLoss"] = "PGDGlobalReconLoss"
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    n_batches: int

class PGDGlobalReconSubsetLossConfig(BaseConfig):
    classname: Literal["PGDGlobalReconSubsetLoss"] = "PGDGlobalReconSubsetLoss"
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    n_batches: int

class PGDGlobalReconLayerwiseLossConfig(BaseConfig):
    classname: Literal["PGDGlobalReconLayerwiseLoss"] = "PGDGlobalReconLayerwiseLoss"
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    n_batches: int
```

**Key differences from PGDConfig:**
- No `mask_scope` field (not needed for global version)
- No `coeff` field (these are not used as loss metrics during training)
- Add `n_batches: int` field to control batch accumulation

**Update MetricConfigType union:**
Add the three new configs to `EvalOnlyMetricConfigType` union.

### 2. Utility Function (`spd/metrics/pgd_utils.py`)

Add a new utility function `pgd_global_masked_recon_loss_update` that:

**Function signature:**
```python
def pgd_global_masked_recon_loss_update(
    *,
    model: ComponentModel,
    dataloader: DataLoader | Iterator,
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    sampling: SamplingType,
    use_delta_component: bool,
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    n_batches: int,
) -> Float[Tensor, ""]
```

**Implementation logic:**
1. Initialize `adv_sources` tensor (similar to current implementation, but need to get shape from first batch)
2. For each PGD step:
   - Initialize gradient accumulator to zero
   - For each of `n_batches` batches from dataloader:
     - Get batch from dataloader using `extract_batch_data(next(dataloader))`
     - Calculate `weight_deltas = model.calc_weight_deltas()`
     - Run forward pass: `target_model_output = model(batch, cache_type="input")`
     - Calculate CI: `ci = model.calc_causal_importances(pre_weight_acts=target_model_output.cache, detach_inputs=False, sampling=sampling)`
     - Compute objective_fn with current adv_sources, CI, weight_deltas, and target_out
     - Compute gradients via autograd
     - **Sum** gradients into accumulator
     - If dataloader runs out before n_batches, raise warning and break
   - Apply gradient update: `adv_sources += step_size * accumulated_grads.sign()`
   - Clamp adv_sources to [0, 1]
3. After all PGD steps, compute final loss with final adv_sources
4. Return final loss

**Key differences from `pgd_masked_recon_loss_update`:**
- Takes dataloader instead of single batch
- Takes individual PGD config fields instead of PGDConfig object
- Does not take pre-computed `ci` or `weight_deltas` - computes them fresh for each batch
- Takes `sampling` and `use_delta_component` to compute CI and weight deltas
- Accumulates gradients over multiple batches before each PGD update
- No mask_scope logic (always uses "unique_per_datapoint" equivalent)
- Only returns loss (not loss and n_examples tuple)

### 3. Metric Functions (`spd/metrics/pgd_global_losses.py` - new file)

Create a new file with three functions that wrap the utility:

**`pgd_global_recon_loss`:**
```python
def pgd_global_recon_loss(
    *,
    model: ComponentModel,
    dataloader: DataLoader | Iterator,
    output_loss_type: Literal["mse", "kl"],
    sampling: SamplingType,
    use_delta_component: bool,
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
    n_batches: int,
) -> Float[Tensor, ""]
```
- Calls utility with `routing="all"`

**`pgd_global_recon_subset_loss`:**
- Same signature as above
- Calls utility with `routing="uniform_k-stochastic"`

**`pgd_global_recon_layerwise_loss`:**
- Same signature as above
- Iterates over each layer (similar to `_pgd_recon_layerwise_loss_update`)
- For each layer, calls utility with `routing="all"` but only for that specific layer
- Sums losses across all layers

### 4. Integration Points

**Where these will be called:**
- These functions will be called manually in evaluation/loss computation code
- Gated by checking if the corresponding config exists in `eval_metric_configs` or `loss_metric_configs`
- Example gating logic:
  ```python
  if any(isinstance(cfg, PGDGlobalReconLossConfig) for cfg in config.eval_metric_configs):
      loss = pgd_global_recon_loss(...)
  ```

**No Metric classes needed:**
These are standalone functions (not Metric classes with update/compute pattern) because they need the full dataloader upfront.

## Implementation Order

1. Add config classes to `spd/configs.py`
2. Implement `pgd_global_masked_recon_loss_update` in `spd/metrics/pgd_utils.py`
3. Create `spd/metrics/pgd_global_losses.py` with three wrapper functions
4. Update config type unions in `spd/configs.py`

## Testing Strategy

- Test with small `n_batches` values to verify gradient accumulation
- Verify warning is raised when dataloader exhausts
- Compare results with single-batch PGD to ensure correctness
- Test all three routing variants (all, subset, layerwise)

## Notes

- These metrics are computationally expensive (run PGD over multiple batches)
- They provide a stronger adversarial evaluation than single-batch PGD
- The gradient accumulation gives PGD more information to find adversarial masks
