# SPD Performance Optimization Opportunities

This document outlines potential speedups for the SPD training pipeline, particularly for
language model decomposition (`lm_decomposition.py`, `run_spd.py`).

## Table of Contents
1. [Data Loading Optimizations](#1-data-loading-optimizations)
2. [torch.compile Opportunities](#2-torchcompile-opportunities)
3. [Memory Transfer Optimizations](#3-memory-transfer-optimizations)
4. [Mixed Precision Training](#4-mixed-precision-training)
5. [Hook Architecture Redesign](#5-hook-architecture-redesign)
6. [Computation Optimizations](#6-computation-optimizations)
7. [Async Operations](#7-async-operations)
8. [Memory Optimizations](#8-memory-optimizations)
9. [Profiling Recommendations](#9-profiling-recommendations)

---

## 1. Data Loading Optimizations

### 1.1 Pin Memory

**Current:** `spd/data.py:256-265` - DataLoader is created without `pin_memory=True`

**Change:**
```python
loader = DataLoader[Dataset | IterableDataset](
    torch_dataset,
    batch_size=batch_size,
    sampler=sampler,
    shuffle=...,
    drop_last=True,
    generator=generator,
    pin_memory=True,  # ADD THIS
    num_workers=4,     # ADD THIS (adjust based on system)
)
```

**Why:** Pinned (page-locked) memory enables faster CPU-to-GPU transfers and is required for
`non_blocking=True` to be effective. With `num_workers > 0`, data loading happens in parallel
processes, overlapping with GPU computation.

**Impact:** Moderate. Most impactful when data loading is a bottleneck.

### 1.2 Prefetch with Background Workers

**Current:** Workers default to 0 (main process loading)

**Change:** Add `num_workers=4` (or experiment with 2-8 depending on CPU cores)

**Caveat:** For tokenized datasets that are already in tensor format, the benefit is smaller. Also
be careful with `IterableDataset` + `num_workers > 0`.

---

## 2. torch.compile Opportunities

### 2.1 Can You Compile the Hook-Based ComponentModel?

**Short answer:** Partially, with caveats.

**The Problem:**
- `ComponentModel.forward()` uses a context manager `_attach_forward_hooks()` that registers and
  removes hooks on every call
- This creates "graph breaks" that prevent full compilation
- The dynamic hook registration pattern is fundamentally incompatible with static graph compilation

**Partial Solutions:**

#### 2.1.1 Compile Inner Modules (RECOMMENDED)

Compile the stateless/pure components that don't use hooks:

```python
# In ComponentModel.__init__ or after creation:
for name in self.target_module_paths:
    self.ci_fns[name] = torch.compile(self.ci_fns[name])
    # Components have weight properties, so be careful:
    # self.components[name] = torch.compile(self.components[name])
```

The `ci_fns` (causal importance functions) are pure MLPs that can be compiled safely.

**Location:** `spd/models/components.py` - `MLPCiFn`, `VectorMLPCiFn`, `VectorSharedMLPCiFn`

#### 2.1.2 Compile the Target Model

```python
# In lm_decomposition.py after loading target_model:
target_model = torch.compile(target_model)  # Full compilation
# OR
target_model = torch.compile(target_model, mode="reduce-overhead")  # Less memory
```

This should work if the target model (e.g., GPT-2) is itself compilable. The hooks will still
work but there may be graph breaks at hook boundaries.

#### 2.1.3 Compile Loss Functions

```python
# In run_spd.py or losses.py:
from spd.metrics import faithfulness_loss, importance_minimality_loss
faithfulness_loss = torch.compile(faithfulness_loss)
importance_minimality_loss = torch.compile(importance_minimality_loss)
```

#### 2.1.4 Full Model Compilation (Experimental)

If you want to try compiling with hooks, use:

```python
model = torch.compile(model, fullgraph=False)  # Allow graph breaks
```

This will compile the parts between graph breaks. Profile to see if it helps.

### 2.2 Redesign for Full Compilation

To enable full `torch.compile`, you'd need to redesign `ComponentModel.forward()` to avoid
dynamic hook registration. See [Section 5](#5-hook-architecture-redesign).

---

## 3. Memory Transfer Optimizations

### 3.1 non_blocking Transfers

**Current:** `spd/run_spd.py:259`
```python
microbatch = extract_batch_data(next(train_iterator)).to(device)
```

**Change:**
```python
microbatch = extract_batch_data(next(train_iterator)).to(device, non_blocking=True)
```

**Requirements:** Only effective when:
1. Source tensor is in pinned memory (via DataLoader `pin_memory=True`)
2. You have operations that can overlap with the transfer

**Additional locations to update:**
- `spd/eval.py:307` - `batch.to(device)`
- `spd/metrics/pgd_utils.py:222` - `microbatch.to(device)`
- All other `.to(device)` calls in the hot path

### 3.2 Overlap Transfer with Computation

With `non_blocking=True`, you can prefetch the next batch while computing:

```python
# In run_spd.py optimize() training loop
next_batch_raw = next(train_iterator)
next_batch = extract_batch_data(next_batch_raw).to(device, non_blocking=True)

for step in tqdm(range(config.steps + 1), ...):
    # Use current batch
    microbatch = next_batch

    # Prefetch next batch (transfer overlaps with forward/backward)
    next_batch_raw = next(train_iterator)
    next_batch = extract_batch_data(next_batch_raw).to(device, non_blocking=True)

    # ... rest of training step
```

---

## 4. Mixed Precision Training (AMP)

**Current:** No mixed precision is used.

**Impact:** Potentially 2x speedup on modern GPUs (Ampere+), especially for large models.

### 4.1 Basic AMP Implementation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step in tqdm(range(config.steps + 1), ...):
    optimizer.zero_grad()

    for _ in range(config.gradient_accumulation_steps):
        microbatch = extract_batch_data(next(train_iterator)).to(device, non_blocking=True)

        with autocast():
            target_model_output = wrapped_model(microbatch, cache_type="input")
            ci = component_model.calc_causal_importances(...)
            microbatch_total_loss, microbatch_loss_terms = compute_total_loss(...)

        scaler.scale(microbatch_total_loss / config.gradient_accumulation_steps).backward()

    scaler.unscale_(optimizer)
    if config.grad_clip_norm_components is not None:
        clip_grad_norm_(component_params, config.grad_clip_norm_components)
    scaler.step(optimizer)
    scaler.update()
```

### 4.2 Considerations

- The `faithfulness_loss` computes `target_weight - components.weight` which should remain in
  full precision for stability
- KL divergence loss may need careful handling of log operations
- Test thoroughly for numerical stability before using

### 4.3 BFloat16 Alternative

For more stability with slightly less speedup:

```python
with autocast(dtype=torch.bfloat16):
    ...
```

No `GradScaler` needed with bfloat16.

---

## 5. Hook Architecture Redesign

The current design registers/removes hooks on every forward pass, which:
1. Prevents `torch.compile` from working fully
2. Adds Python overhead

### 5.1 Persistent Hooks with Closure Variables

Instead of context-managed hooks, register hooks once and control behavior via closure:

```python
class ComponentModel(LoadableModule):
    def __init__(self, ...):
        ...
        self._hook_state: dict[str, Any] = {}
        self._register_persistent_hooks()

    def _register_persistent_hooks(self):
        """Register hooks once during initialization."""
        for module_name in self.target_module_paths:
            target_module = self.target_model.get_submodule(module_name)
            target_module.register_forward_hook(
                partial(self._persistent_hook, module_name=module_name),
                with_kwargs=True,
            )

    def _persistent_hook(self, module, args, kwargs, output, module_name):
        """Hook that checks closure state to determine behavior."""
        state = self._hook_state.get(module_name)
        if state is None:
            return None  # No-op

        # ... implement based on state

    def forward(self, *args, mask_infos=None, cache_type="none", **kwargs):
        # Set state before forward
        for name in self.target_module_paths:
            self._hook_state[name] = {
                "mask_info": mask_infos[name] if mask_infos else None,
                "cache_type": cache_type,
                "cache": {},
            }

        raw_out = self.target_model(*args, **kwargs)

        # Gather cached values
        cache = {name: self._hook_state[name]["cache"]
                 for name in self.target_module_paths}

        # Clear state
        self._hook_state.clear()

        return ...
```

**Benefit:** Removes per-forward-pass hook registration overhead.

### 5.2 Alternative: No-Hook Architecture

For maximum performance, compute component activations explicitly without hooks:

```python
def forward_with_components(self, x, mask_infos):
    """Explicit forward pass computing component activations inline."""
    # This requires knowing the model architecture
    # Example for transformer:
    hidden = self.target_model.embed(x)

    for layer_idx, layer in enumerate(self.target_model.layers):
        # MLP
        mlp_input = layer.ln1(hidden)
        mlp_input_cached = mlp_input  # Cache for CI

        if mask_infos:
            mlp_out = self.components[f"layer.{layer_idx}.mlp"](
                mlp_input, mask=mask_infos[...].component_mask
            )
        else:
            mlp_out = layer.mlp(mlp_input)

        hidden = hidden + mlp_out
        # ... attention, etc.

    return hidden
```

**Caveat:** This is a significant rewrite and loses generality.

---

## 6. Computation Optimizations

### 6.1 Avoid Redundant Weight Delta Computation

**Current:** `spd/run_spd.py:254` computes `weight_deltas` every step before gradient accumulation

```python
weight_deltas = component_model.calc_weight_deltas()
```

This computes `target_weight - V @ U` for all layers. The target weights are frozen, so only
`V @ U` changes.

**Optimization:** Cache target weights and only recompute `V @ U`:

```python
# In ComponentModel:
@functools.lru_cache(maxsize=1)
def _cached_target_weights(self) -> dict[str, Tensor]:
    return {name: self.target_weight(name) for name in self.target_module_paths}

def calc_weight_deltas(self) -> dict[str, Tensor]:
    target_weights = self._cached_target_weights()
    return {name: target_weights[name] - self.components[name].weight
            for name in self.target_module_paths}
```

### 6.2 Fused einsum Operations

**Current:** `spd/models/components.py` uses separate einsum calls

```python
component_acts = self.get_inner_acts(x)  # x @ V
out = einops.einsum(component_acts, self.U, ...)  # inner @ U
```

**Alternative:** If mask is None (common in target model forward), fuse:

```python
if mask is None and weight_delta_and_mask is None:
    # Fused path: x @ (V @ U)
    return x @ self.weight.T + (self.bias if self.bias is not None else 0)
```

### 6.3 Reduce Dictionary Overhead

Python dict operations have overhead. For hot paths, consider:

```python
# Instead of:
for target_module_name in pre_weight_acts:
    ...

# Use ordered lists with known indices:
for idx, (name, acts) in enumerate(zip(self.target_module_paths, pre_weight_acts_list)):
    ...
```

### 6.4 PGD Optimization

The PGD loss (`pgd_recon_loss`, `pgd_recon_subset_loss`) runs multiple forward passes per step.

**Current:** `gpt2_config.yaml` has `n_steps: 20` for PGD

Consider:
- Reducing `n_steps` during early training when masks are far from optimal
- Using larger `step_size` with fewer steps
- Running PGD only on subset of batches

---

## 7. Async Operations

### 7.1 Async Checkpointing

**Current:** `spd/run_spd.py:389` saves synchronously

```python
save_file(component_model.state_dict(), out_dir / f"model_{step}.pth")
```

**Change:** Save in background thread:

```python
import threading
from queue import Queue

class AsyncSaver:
    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            state_dict, path = self.queue.get()
            if state_dict is None:
                break
            torch.save(state_dict, path)
            self.queue.task_done()

    def save(self, state_dict, path):
        # Copy to CPU first to avoid blocking GPU
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        self.queue.put((cpu_state, path))

# Usage:
async_saver = AsyncSaver()
async_saver.save(component_model.state_dict(), out_dir / f"model_{step}.pth")
```

### 7.2 Async Logging

`wandb.log` can be slow. Use `commit=False` and batch logs:

```python
# Instead of:
wandb.log(microbatch_log_data, step=step)

# Use:
wandb.log(microbatch_log_data, step=step, commit=False)
# Then commit at eval or every N steps:
wandb.log({}, commit=True)
```

---

## 8. Memory Optimizations

### 8.1 Reduce gc.collect() Calls

**Current:** `spd/run_spd.py:95-96` and `375-377` call:
```python
torch.cuda.empty_cache()
gc.collect()
```

These are expensive. Consider:
1. Only calling after OOM errors
2. Calling less frequently
3. The order should be `gc.collect()` then `torch.cuda.empty_cache()`

### 8.2 Gradient Checkpointing

For very large models, enable gradient checkpointing:

```python
# For HuggingFace models:
target_model.gradient_checkpointing_enable()
```

**Tradeoff:** ~30% slower but uses significantly less memory, allowing larger batch sizes.

### 8.3 Delete Intermediate Tensors

Explicitly delete large intermediates:

```python
# After computing loss:
del target_model_output
del ci
# Let PyTorch reclaim memory for next iteration
```

---

## 9. Profiling Recommendations

Before implementing changes, profile to find actual bottlenecks:

### 9.1 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        # ... training step
        prof.step()
        if step >= 5:
            break
```

### 9.2 CUDA Events for Timing

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... operation
end.record()
torch.cuda.synchronize()
print(f"Operation took {start.elapsed_time(end)} ms")
```

### 9.3 Simple Timing

```python
import time

torch.cuda.synchronize()
t0 = time.perf_counter()
# ... operation
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"Operation took {(t1-t0)*1000:.2f} ms")
```

---

## Priority Recommendations

Based on likely impact vs. implementation effort:

### High Impact, Low Effort
1. **Pin memory + num_workers** in DataLoader
2. **non_blocking=True** for `.to(device)` calls
3. **torch.compile ci_fns** (the causal importance MLPs)
4. **Async checkpointing**

### High Impact, Medium Effort
5. **Mixed precision training** (AMP or bfloat16)
6. **torch.compile target_model**
7. **Reduce gc.collect() frequency**

### Medium Impact, High Effort
8. **Persistent hook architecture**
9. **Prefetch batches with overlap**

### Profile First
10. **PGD optimizations** (depends on how much time PGD takes)
11. **Fused operations** (may already be handled by torch.compile)

---

## Quick Start Checklist

Minimal changes for immediate speedup:

```python
# 1. In spd/data.py create_data_loader():
loader = DataLoader(..., pin_memory=True, num_workers=4)

# 2. In spd/run_spd.py optimize():
microbatch = extract_batch_data(next(train_iterator)).to(device, non_blocking=True)

# 3. In spd/models/component_model.py __init__():
for name in self.target_module_paths:
    self.ci_fns[name] = torch.compile(self.ci_fns[name], mode="reduce-overhead")

# 4. In lm_decomposition.py after loading target_model:
target_model = torch.compile(target_model)
```
