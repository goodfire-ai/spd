# GPU Memory Leak Analysis for LM Decomposition

## Summary of Findings

Based on my analysis of the codebase, I've identified several potential sources of GPU memory leaks when running `python spd/experiments/lm/lm_decomposition.py spd/experiments/lm/ss_mlp_config.yaml` with batch size 1.

## Key Observations from the Screenshot

1. **Gradual Step-wise Increase**: Memory usage increases from ~4% to ~13% in discrete steps
2. **Irregular Intervals**: The increases don't happen at regular time intervals
3. **No Memory Recovery**: Memory never decreases, suggesting persistent tensor retention
4. **Small Increments**: Each step is roughly 1-2% of GPU memory

## Identified Potential Memory Leak Sources

### 1. **Evaluation Metric Accumulation** (MOST LIKELY)

In `spd/eval.py`, several `StreamingEval` classes accumulate data across evaluation steps:

- **`CIHistograms`** (line 215-216): 
  ```python
  self.causal_importances[k].append(v.detach().cpu())
  ```
  This appends tensors to lists that grow unbounded during evaluation.

- **`ComponentActivationDensity`** (line 252):
  ```python
  self.component_activation_counts[module_name] += n_activations_per_component
  ```
  Accumulates counts but the tensors remain on GPU.

### 2. **Multiple Forward Passes in Evaluation**

In `CEandKLLosses._calc_ce_and_kl_losses` (eval.py:102-195), there are **6 different forward passes**:
- CI masked
- Stochastic masked  
- Unmasked
- Random masked
- Rounded masked
- Zero masked

Each forward pass creates new output tensors. With batch_size=1 and max_seq_len=512, this creates multiple 512-length sequences of logits.

### 3. **Hook-based Caching**

In `ComponentModel.forward_with_pre_forward_cache_hooks` (component_model.py:307-344):
- Creates a `cache` dictionary that stores input tensors
- While hooks are removed, the cache is returned and might be retained in the optimization loop

### 4. **AliveComponentsTracker State**

The `AliveComponentsTracker` maintains persistent state across the entire training run:
```python
self.examples_since_fired[module_name] = torch.where(
    firing,
    0, 
    self.examples_since_fired[module_name] + n_examples,
)
```

### 5. **Stochastic Mask Generation**

During loss calculation, multiple stochastic masks are generated:
- `calc_stochastic_masks` is called multiple times per step
- Each call creates new mask tensors that might not be properly freed

## Why the Sporadic Pattern?

The memory increases align with evaluation frequency:
- `eval_freq = 1000` - Regular evaluation every 1000 steps
- `slow_eval_freq = 5000` - Slow evaluation (with visualizations) every 5000 steps
- `slow_eval_on_first_step = true` - Slow eval also runs on step 0

The irregular timing in the graph suggests memory accumulation happens primarily during evaluation phases.

## Recommendations for Fixing

1. **Clear Evaluation Metric Accumulators**: 
   - In `CIHistograms`, limit the list size or periodically clear old data
   - Move tensors to CPU more aggressively and delete GPU references

2. **Optimize Multiple Forward Passes**:
   - Consider batching the different mask types in a single forward pass
   - Explicitly delete intermediate outputs after use

3. **Fix Hook-based Caching**:
   - Ensure the cache dictionary is not retained after use
   - Consider using `with torch.no_grad()` more extensively

4. **Periodic Memory Cleanup**:
   - Add `torch.cuda.empty_cache()` after evaluation
   - Force garbage collection after heavy computation phases

5. **Limit Visualization Data**:
   - The plotting functions in slow eval might retain references to GPU tensors
   - Ensure all data is moved to CPU before plotting

## Testing Recommendations

1. Run with `slow_eval_freq = None` to disable slow evaluation and see if the leak persists
2. Temporarily disable specific eval metrics in the config to isolate which one causes the leak
3. Add memory profiling hooks to track exactly when memory increases occur
4. Use PyTorch's memory profiler to get detailed allocation traces