# Plan: Batch Autograd Optimization for Global Attributions

## Goal
Reduce computation time by batching multiple input layers into a single `autograd.grad` call per output layer, instead of one call per (input, output) pair.

## Key Insight
For each output layer, we can compute gradients w.r.t. all its input layers in one backward pass. This shares intermediate gradient computations.

**Structural property we rely on:**
- `o_proj` outputs: ALL inputs are attention pairs (q/k/v from same block only)
- Non-`o_proj` outputs: NO inputs are attention pairs

This means we never have mixed attention/non-attention inputs for the same output.

## Changes

### 1. Refactor `get_valid_pairs` → `get_sources_by_target`

**File:** `scripts/calc_global_attributions.py`
**Lines:** 166-250

Change return type from `list[tuple[str, str]]` to `dict[str, list[str]]`.

```python
def get_sources_by_target(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    n_blocks: int,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

    Returns:
        Dict mapping out_layer -> list of in_layers that have gradient flow to it.
    """
```

Build the dict directly instead of a list of pairs:
```python
sources_by_target: dict[str, list[str]] = defaultdict(list)
# ... existing gradient checking logic ...
if has_grad:
    sources_by_target[out_layer].append(in_layer)
return dict(sources_by_target)
```

### 2. Add validation function

**Location:** After `get_sources_by_target`, before `compute_global_attributions`

```python
def validate_attention_pair_structure(sources_by_target: dict[str, list[str]]) -> None:
    """Assert that o_proj layers only receive from same-block QKV.

    This structural property allows us to handle attention and non-attention
    cases separately without mixing.
    """
    for out_layer, in_layers in sources_by_target.items():
        if "o_proj" in out_layer:
            out_block = out_layer.split(".")[1]
            for in_layer in in_layers:
                assert any(x in in_layer for x in ["q_proj", "k_proj", "v_proj"]), \
                    f"o_proj output {out_layer} has non-QKV input {in_layer}"
                in_block = in_layer.split(".")[1]
                assert in_block == out_block, \
                    f"o_proj output {out_layer} has input from different block: {in_layer}"
        else:
            for in_layer in in_layers:
                assert not is_qkv_to_o_pair(in_layer, out_layer), \
                    f"Non-o_proj output {out_layer} has attention pair input {in_layer}"
```

### 3. Refactor `compute_global_attributions`

**File:** `scripts/calc_global_attributions.py`
**Lines:** 253-431

#### 3.1 Change signature

```python
def compute_global_attributions(
    model: ComponentModel,
    data_loader: Iterable[dict[str, Any]],
    device: str,
    config: Config,
    sources_by_target: dict[str, list[str]],  # Changed from valid_pairs
    max_batches: int,
    alive_indices: dict[str, list[int]],
) -> dict[tuple[str, str], Tensor]:
```

#### 3.2 Change main loop structure

From:
```python
for in_layer, out_layer in tqdm(valid_pairs, desc="Layer pairs", leave=False):
    # ... one autograd call ...
```

To:
```python
for out_layer, in_layers in tqdm(sources_by_target.items(), desc="Target layers", leave=False):
    # ... one autograd call for ALL in_layers ...
```

#### 3.3 Batch the autograd call

```python
# Gather all input tensors for this output layer
in_tensors = [cache[f"{in_layer}_post_detach"] for in_layer in in_layers]

# Single autograd call with multiple inputs
grads_tuple = torch.autograd.grad(
    outputs=out_pre_detach,
    inputs=in_tensors,
    grad_outputs=batched_grad_outputs,
    retain_graph=True,
    is_grads_batched=True,
)
```

#### 3.4 Process results per input layer

```python
for i, in_layer in enumerate(in_layers):
    grads = grads_tuple[i]  # [n_alive_out, B, S, C]
    alive_in = alive_indices[in_layer]
    ci_in = ci.lower_leaky[in_layer]
    ci_weighted_in = in_tensors[i] * ci_in

    # Same weighted gradient logic as before
    weighted = grads * ci_weighted_in.unsqueeze(0)
    weighted_alive = weighted[:, :, :, alive_in]
    batch_attribution = weighted_alive.pow(2).sum(dim=(1, 2)).T

    attribution_sums[(in_layer, out_layer)] += batch_attribution
    samples_per_pair[(in_layer, out_layer)] += n_samples
```

#### 3.5 Attention vs non-attention handling

```python
is_attention_output = "o_proj" in out_layer

if is_attention_output:
    for s_out in tqdm(range(n_seq), desc="Positions", leave=False):
        batched_grad_outputs = build_batched_grad_outputs(ci_out, alive_out, s_out=s_out)
        grads_tuple = torch.autograd.grad(
            outputs=out_pre_detach,
            inputs=in_tensors,
            grad_outputs=batched_grad_outputs,
            retain_graph=True,
            is_grads_batched=True,
        )

        n_samples = batch_size  # One position contributes batch_size samples
        for i, in_layer in enumerate(in_layers):
            # Process with causal masking: grads[:, :, :s_out+1, :]
            ...
else:
    batched_grad_outputs = build_batched_grad_outputs(ci_out, alive_out)
    grads_tuple = torch.autograd.grad(
        outputs=out_pre_detach,
        inputs=in_tensors,
        grad_outputs=batched_grad_outputs,
        retain_graph=True,
        is_grads_batched=True,
    )

    n_samples = batch_size * n_seq
    for i, in_layer in enumerate(in_layers):
        # Process all positions
        ...
```

### 4. Update call sites

**Lines:** ~493-494, ~522-530

```python
# Before
valid_pairs = get_valid_pairs(model, data_loader, device, config, n_blocks)

# After
sources_by_target = get_sources_by_target(model, data_loader, device, config, n_blocks)
validate_attention_pair_structure(sources_by_target)
```

```python
# Before
global_attributions = compute_global_attributions(
    ...
    valid_pairs=valid_pairs,
    ...
)

# After
global_attributions = compute_global_attributions(
    ...
    sources_by_target=sources_by_target,
    ...
)
```

### 5. Update summary printing

```python
# Before
print(f"Valid layer pairs: {valid_pairs}")

# After
n_pairs = sum(len(ins) for ins in sources_by_target.values())
print(f"Sources by target: {n_pairs} pairs across {len(sources_by_target)} target layers")
for out_layer, in_layers in sources_by_target.items():
    print(f"  {out_layer} <- {in_layers}")
```

## Expected Speedup

For 1-block model with 6 layers:
- Target layers: 5 (everything except q_proj which has no inputs)
- Before: 15 autograd calls per batch (one per pair)
- After: 5 autograd calls per batch (one per target)

For attention (o_proj): Same number of calls but shared backward for q/k/v inputs.

## Testing

1. Run on 1-block model, compare output tensors to ensure correctness
2. Compare wall-clock time before/after
3. Run on multi-block model to verify assertion holds
