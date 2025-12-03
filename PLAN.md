# Plan: Batched CI Sets in calc_local_attributions.py

## Goal

Modify `calc_local_attributions.py` to support multiple sets of `ci_lower_leaky` values, computing attributions for all sets in a single batched forward pass and using the sum-over-batch gradient trick for efficient backward passes.

## Input/Output Changes

### Input JSON Format (New)
```json
{
  "prompt": "They walked hand in",
  "ci_sets": {
    "seed_0": {
      "layer0": [[...], [...], ...],  // [seq][C]
      "layer1": [[...], [...], ...]
    },
    "seed_1": {
      "layer0": [[...], [...], ...],
      "layer1": [[...], [...], ...]
    }
  }
}
```

### Output Type Change
```python
# Old
def compute_local_attributions(...) -> tuple[list[PairAttribution], Tensor]:

# New
def compute_local_attributions(...) -> tuple[dict[str, list[PairAttribution]], Tensor]:
```

Returns `dict[set_name, list[PairAttribution]]` where each set has its own trimmed indices based on that set's alive components.

## Implementation Steps

### Step 1: Update `load_ci_from_json`

**Current signature:**
```python
def load_ci_from_json(ci_vals_path, expected_prompt, device) -> dict[str, Tensor]
```

**New signature:**
```python
def load_ci_from_json(ci_vals_path, expected_prompt, device) -> tuple[
    dict[str, Float[Tensor, "N seq C"]],  # Stacked CI tensors (batch dim = N sets)
    list[str],  # Set names in batch order
]
```

**Implementation:**
1. Parse JSON with new `ci_sets` structure
2. Validate all sets have same layers and shapes
3. Stack tensors along new batch dimension
4. Return stacked tensors + ordered list of set names

### Step 2: Update `compute_layer_alive_info`

**Current:** Assumes batch dim 1, returns single `LayerAliveInfo`

**New:** Handle batch dim N, return per-set alive info

```python
def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],  # Now [N, seq, C]
    output_probs: Tensor | None,        # Now [N, seq, vocab]
    ci_threshold: float,
    output_prob_threshold: float,
    n_seq: int,
    n_batch: int,  # NEW
    device: str,
) -> list[LayerAliveInfo]:  # One per batch item
```

**Implementation:**
- Compute `alive_mask` for full batch: `[N, seq, C]`
- For each batch index, compute that batch's `alive_c_idxs` and `c_to_trimmed`
- Return list of `LayerAliveInfo`, one per batch item

### Step 3: Update `compute_local_attributions` - Setup

**Changes to function signature:**
```python
def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],  # Still [1, seq] - we expand internally
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
    ci_lower_leaky: dict[str, Float[Tensor, "N seq C"]],  # Now required, batched
    set_names: list[str],  # NEW: ordered set names
) -> tuple[dict[str, list[PairAttribution]], Float[Tensor, "N seq vocab"]]:
```

**Setup changes:**
1. Determine `n_batch` from CI tensor shapes
2. Expand tokens: `tokens = tokens.expand(n_batch, -1)` → `[N, seq]`
3. Forward pass now processes all N sets at once
4. `output_probs` becomes `[N, seq, vocab]`

### Step 4: Update `compute_local_attributions` - Alive Info

**Current:**
```python
alive_info: dict[str, LayerAliveInfo] = {}
for layer in all_layers:
    alive_info[layer] = compute_layer_alive_info(...)
```

**New:**
```python
alive_info: dict[str, list[LayerAliveInfo]] = {}  # layer -> [per-batch info]
for layer in all_layers:
    alive_info[layer] = compute_layer_alive_info(...)  # returns list
```

Also compute union of alive components for determining which gradients to compute:
```python
alive_c_union: dict[str, list[int]] = {}  # layer -> union of alive_c_idxs across batches
for layer in all_layers:
    all_alive = set()
    for batch_info in alive_info[layer]:
        all_alive.update(batch_info.alive_c_idxs)
    alive_c_union[layer] = sorted(all_alive)
```

### Step 5: Update `compute_local_attributions` - Gradient Loop

**Current inner loop:**
```python
for s_out in range(n_seq):
    s_out_alive_c = [c for c in target_info.alive_c_idxs if target_info.alive_mask[0, s_out, c]]
    for c_out in s_out_alive_c:
        in_post_detach_grads = torch.autograd.grad(
            outputs=out_pre_detach[0, s_out, c_out],
            inputs=in_post_detaches,
            retain_graph=True,
        )
        # ... store attributions
```

**New inner loop:**
```python
for s_out in range(n_seq):
    # Union of alive c_out across all batches at this position
    s_out_alive_c_union = [
        c for c in alive_c_union[target]
        if any(info.alive_mask[s_out, c] for info in target_infos)
    ]

    for c_out in s_out_alive_c_union:
        # Sum over batch dimension for efficient batched gradient
        in_post_detach_grads = torch.autograd.grad(
            outputs=out_pre_detach[:, s_out, c_out].sum(),  # sum over batch
            inputs=in_post_detaches,
            retain_graph=True,
        )
        # Grads now have shape [N, seq, dim] - per-batch gradients

        # Store attributions per-batch
        for b in range(n_batch):
            # Only store if c_out is alive in this batch at this position
            if not target_infos[b].alive_mask[s_out, c_out]:
                continue
            # ... store in batch b's attribution tensor
```

### Step 6: Update Attribution Storage

**Current:** Single attribution tensor per source-target pair
```python
attributions: list[Tensor] = [
    torch.zeros(n_seq, len(source_info.alive_c_idxs), n_seq, len(target_info.alive_c_idxs), ...)
    for source_info in source_infos
]
```

**New:** Per-batch attribution tensors
```python
# attributions[source_idx][batch_idx] = tensor
attributions: list[list[Tensor]] = [
    [
        torch.zeros(
            n_seq,
            len(source_infos[b].alive_c_idxs),
            n_seq,
            len(target_infos[b].alive_c_idxs),
            device=device,
        )
        for b in range(n_batch)
    ]
    for source_idx in range(len(sources))
]
```

When storing attributions, need to map from original component index to each batch's trimmed index:
```python
for b in range(n_batch):
    if c_out not in target_infos[b].c_to_trimmed:
        continue  # c_out not alive in this batch
    trimmed_c_out = target_infos[b].c_to_trimmed[c_out]
    # ... similarly for c_in
    attributions[source_idx][b][s_in, trimmed_c_in, s_out, trimmed_c_out] = value
```

### Step 7: Build Output Dictionary

**Current:**
```python
local_attributions: list[PairAttribution] = []
# ... append PairAttribution objects
return local_attributions, output_probs
```

**New:**
```python
# Collect per-batch results
local_attributions_by_set: dict[str, list[PairAttribution]] = {
    name: [] for name in set_names
}

for source, source_infos, attr_list in zip(sources, all_source_infos, attributions):
    for b, set_name in enumerate(set_names):
        local_attributions_by_set[set_name].append(
            PairAttribution(
                source=source,
                target=target,
                attribution=attr_list[b],
                trimmed_c_in_idxs=source_infos[b].alive_c_idxs,
                trimmed_c_out_idxs=target_infos[b].alive_c_idxs,
                is_kv_to_o_pair=is_kv_to_o_pair(source, target),
                original_alive_mask_in=original_source_infos[b].alive_mask,
                original_alive_mask_out=original_target_infos[b].alive_mask,
            )
        )

return local_attributions_by_set, output_probs
```

### Step 8: Update `main()` Function

1. Update call to `load_ci_from_json` to get `(ci_lower_leaky, set_names)`
2. Pass `set_names` to `compute_local_attributions`
3. Handle dict output for saving/plotting

**Saving:** Either:
- Save single `.pt` file with dict structure
- Save separate files per set (e.g., `local_attributions_{wandb_id}_{set_name}.pt`)

**Plotting:** Loop over sets and generate separate plots:
```python
for set_name, attr_pairs in attr_pairs_by_set.items():
    fig = plot_local_graph(attr_pairs=attr_pairs, ...)
    fig.savefig(f"local_attribution_graph_{wandb_id}_{set_name}.png", ...)
```

## Summary of Key Changes

| Component | Current | New |
|-----------|---------|-----|
| `ci_lower_leaky` shape | `[1, seq, C]` | `[N, seq, C]` |
| `tokens` shape | `[1, seq]` | `[N, seq]` (expanded) |
| `alive_info` type | `dict[str, LayerAliveInfo]` | `dict[str, list[LayerAliveInfo]]` |
| Gradient output | `out_pre_detach[0, s_out, c_out]` | `out_pre_detach[:, s_out, c_out].sum()` |
| Attribution storage | `list[Tensor]` | `list[list[Tensor]]` (per-batch) |
| Return type | `list[PairAttribution]` | `dict[str, list[PairAttribution]]` |

## Edge Cases to Handle

1. **Single CI set**: Should still work — dict with one key, batch dim N=1
2. **Different alive components per set**: Handled by per-batch `LayerAliveInfo` and trimmed indices
3. **Original CI tracking**: Compute once from model (not batched), then create per-batch `original_alive_info` by repeating (or compute separately if needed)

## Testing

1. Run with single CI set JSON → verify output matches current behavior (modulo dict wrapper)
2. Run with multiple CI sets → verify each set's attributions are correct
3. Compare batched results vs sequential single-set runs → should match numerically
