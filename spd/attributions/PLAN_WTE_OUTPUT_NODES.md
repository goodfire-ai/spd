# Plan: Add WTE (Input Embedding) and Output Logit Nodes to Attribution Graphs

## Overview

Your colleague's branch (`feature/global-attr`) has implemented support for:
1. **WTE (Word Token Embeddings)** - Input layer showing how token embeddings flow into the first transformer block
2. **Output Logits** - Output layer showing how the final transformer block feeds into predicted tokens

These features are in `spd/scripts/calc_local_attributions.py` and `spd/scripts/calc_global_attributions.py` on `feature/global-attr`. We need to port them to the `spd/attributions/` module on `feature/local-attr`.

## Current State

**On `feature/local-attr` (your branch):**
- `spd/attributions/compute.py` has `compute_local_attributions()` but NO wte/output support
- `spd/attributions/generate.py` generates databases
- `spd/attributions/serve.py` serves the API
- `spd/attributions/db.py` handles storage
- `spd/scripts/local_attributions.html` is the frontend

**On `feature/global-attr` (colleague's branch):**
- `spd/scripts/calc_local_attributions.py` has FULL wte/output support
- `spd/scripts/calc_global_attributions.py` has matching support
- `spd/scripts/plot_local_attributions.py` handles visualization
- NO `spd/attributions/` module exists

## Detailed Implementation Plan

### Step 1: Update `spd/attributions/compute.py`

#### 1.1 Add `LayerAliveInfo` dataclass

```python
@dataclass
class LayerAliveInfo:
    """Info about alive components for a layer."""
    alive_mask: Bool[Tensor, "1 s dim"]  # Which (pos, component) pairs are alive
    alive_c_idxs: list[int]              # Components alive at any position
    c_to_trimmed: dict[int, int]         # original idx -> trimmed idx
```

#### 1.2 Add `compute_layer_alive_info()` function

This handles the THREE types of layers differently:

```python
def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],
    output_probs: Float[Tensor, "1 s vocab"] | None,
    ci_threshold: float,
    output_prob_threshold: float,
    n_seq: int,
    device: str,
) -> LayerAliveInfo:
    """Compute alive info for a layer. Handles regular, wte, and output layers."""

    if layer_name == "wte":
        # WTE: single pseudo-component, always alive at all positions
        # We collapse the entire embedding dimension into one "component"
        alive_mask = torch.ones(1, n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]

    elif layer_name == "output":
        # OUTPUT: each vocab token is a "component", alive if prob >= threshold
        assert output_probs is not None
        alive_mask = output_probs >= output_prob_threshold  # [1, s, vocab]
        alive_c_idxs = torch.where(alive_mask[0].any(dim=0))[0].tolist()

    else:
        # REGULAR: use CI threshold
        ci = ci_lower_leaky[layer_name]
        alive_mask = ci >= ci_threshold
        alive_c_idxs = torch.where(alive_mask[0].any(dim=0))[0].tolist()

    c_to_trimmed = {c: i for i, c in enumerate(alive_c_idxs)}
    return LayerAliveInfo(alive_mask, alive_c_idxs, c_to_trimmed)
```

#### 1.3 Update `get_sources_by_target()` to include wte and output

Current version only includes `h.{i}.{sublayer}` layers. Need to:

1. Add "wte" to the start of the layer list
2. Add "output" to the end of the layer list
3. Add a hook to capture `wte` embeddings with gradients
4. Store `output_pre_detach` in cache

```python
def get_sources_by_target(
    model: ComponentModel,
    device: str,
    sampling: str,
    n_blocks: int,
) -> dict[str, list[str]]:
    # ... existing setup ...

    # NEW: Hook to capture wte output with gradients
    wte_cache: dict[str, Tensor] = {}

    def wte_hook(_module: nn.Module, _args: Any, _kwargs: Any, output: Tensor) -> Any:
        output.requires_grad_(True)
        wte_cache["wte_post_detach"] = output
        return output

    assert isinstance(model.target_model.wte, nn.Module)
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    # Forward pass with grad enabled
    with torch.enable_grad():
        comp_output_with_cache = model(batch, mask_infos=mask_infos, cache_type="component_acts")

    wte_handle.remove()

    # Add wte and output to cache
    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cache["wte_post_detach"]
    cache["output_pre_detach"] = comp_output_with_cache.output

    # NEW: Build layer list with wte at start, output at end
    layers = ["wte"]
    component_layers = ["attn.q_proj", "attn.k_proj", "attn.v_proj",
                        "attn.o_proj", "mlp.c_fc", "mlp.down_proj"]
    for i in range(n_blocks):
        layers.extend([f"h.{i}.{name}" for name in component_layers])
    layers.append("output")

    # Test pairs: wte can feed into anything, anything can feed into output
    test_pairs = []
    for in_layer in layers[:-1]:  # Don't include "output" as source
        for out_layer in layers[1:]:  # Don't include "wte" as target
            if layers.index(in_layer) < layers.index(out_layer):
                test_pairs.append((in_layer, out_layer))

    # ... rest of gradient testing ...
```

#### 1.4 Update `compute_local_attributions()`

Major changes needed:

**A. Add wte hook and output caching:**
```python
def compute_local_attributions(...):
    # ... existing CI computation ...

    # NEW: Hook to capture wte
    wte_cache: dict[str, Tensor] = {}

    def wte_hook(_module, _args, _kwargs, output):
        output.requires_grad_(True)
        wte_cache["wte_post_detach"] = output
        return output

    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    # Forward with component acts
    with torch.enable_grad():
        comp_output_with_cache = model(tokens, mask_infos=mask_infos, cache_type="component_acts")

    wte_handle.remove()

    # Add to cache
    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cache["wte_post_detach"]
    cache["output_pre_detach"] = comp_output_with_cache.output

    # Compute output probs for thresholding
    output_probs = torch.softmax(comp_output_with_cache.output, dim=-1)
```

**B. Compute alive info for ALL layers including wte/output:**
```python
    # Collect all layers from sources_by_target
    all_layers: set[str] = set(sources_by_target.keys())
    for sources in sources_by_target.values():
        all_layers.update(sources)

    alive_info: dict[str, LayerAliveInfo] = {}
    for layer in all_layers:
        alive_info[layer] = compute_layer_alive_info(
            layer, ci.lower_leaky, output_probs,
            ci_threshold, output_prob_threshold, n_seq, device
        )
```

**C. Handle wte specially in gradient computation:**
```python
    # In the gradient loop:
    for source, source_info, grad, in_post_detach, attr in zip(...):
        weighted = (grad * in_post_detach)[0]

        # NEW: WTE has no components - sum over embedding dim
        if source == "wte":
            weighted = weighted.sum(dim=1, keepdim=True)  # [s, 1]

        # ... rest of attribution accumulation ...
```

#### 1.5 Add new parameters

Add these to `compute_local_attributions()`:
- `output_prob_threshold: float` - threshold for considering output tokens alive (e.g., 0.1)

### Step 2: Update `spd/attributions/generate.py`

#### 2.1 Add `output_prob_threshold` to `GenerateConfig`

```python
@dataclass
class GenerateConfig:
    # ... existing fields ...
    output_prob_threshold: float = 0.1  # NEW
```

#### 2.2 Update worker function to pass new parameter

```python
attr_pairs = compute_local_attributions(
    model=model,
    tokens=batch,
    sources_by_target=sources_by_target,
    ci_threshold=config.ci_threshold,
    output_prob_threshold=config.output_prob_threshold,  # NEW
    sampling=spd_config.sampling,
    device=device,
)
```

#### 2.3 Update `extract_active_components()` to handle wte/output

```python
def extract_active_components(attr_pairs: list[PairAttribution]) -> dict[str, ComponentActivation]:
    """Extract active components from attribution pairs for the inverted index."""
    active: dict[str, ComponentActivation] = {}

    for pair in attr_pairs:
        # Handle source layer
        for c_idx in pair.trimmed_c_in_idxs:
            # NEW: wte has single pseudo-component
            if pair.source == "wte":
                key = "wte:0"  # Always component 0
            else:
                key = f"{pair.source}:{c_idx}"
            # ... rest of accumulation ...

        # Handle target layer
        for c_idx in pair.trimmed_c_out_idxs:
            # NEW: output components are vocab token indices
            if pair.target == "output":
                key = f"output:{c_idx}"  # c_idx is vocab token ID
            else:
                key = f"{pair.target}:{c_idx}"
            # ... rest of accumulation ...
```

### Step 3: Update Database Schema (Optional Enhancement)

Consider adding output token labels to the database:

```python
# In db.py, add to meta table:
db.set_meta("vocab_labels", {token_id: tokenizer.decode([token_id]) for token_id in alive_output_ids})
```

This allows the frontend to display token strings for output nodes.

### Step 4: Update Frontend (`local_attributions.html`)

#### 4.1 Add colors for new layer types

```javascript
const COLORS = {
    // ... existing colors ...
    wte: "#34495E",      // Dark blue-gray for embeddings
    output: "#1ABC9C",   // Teal for output logits
};
```

#### 4.2 Update `parseLayer()` to handle wte/output

```javascript
function parseLayer(name) {
    if (name === "wte") return { name, block: -1, type: "embed", subtype: "wte" };
    if (name === "output") return { name, block: 999, type: "output", subtype: "output" };

    const m = name.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
    return m ? { name, block: +m[1], type: m[2], subtype: m[3] } : null;
}
```

#### 4.3 Update layer ordering

```javascript
const SUBTYPE_ORDER = [
    'wte',        // NEW: First
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'c_fc', 'down_proj',
    'output'      // NEW: Last
];
```

#### 4.4 Display output token labels

When hovering over output nodes, show the decoded token string instead of just the index:
```javascript
// In node tooltip:
if (layer === "output") {
    const tokenLabel = outputTokenLabels[cIdx] || `token_${cIdx}`;
    infoEl.innerHTML += `<p><strong>Output Token:</strong> "${tokenLabel}"</p>`;
}
```

### Step 5: Update Activation Contexts

The activation contexts currently only cover component layers. For wte/output:

- **wte**: No activation context needed (single pseudo-component, always active)
- **output**: Could add top predicted tokens per position, but this is per-prompt not per-component

Consider adding a new field to the prompt data:
```python
{
    "tokens": [...],
    "pairs": [...],
    "output_token_labels": {123: "the", 456: " cat", ...}  # NEW
}
```

## File-by-File Changes Summary

| File | Changes |
|------|---------|
| `spd/attributions/compute.py` | Add `LayerAliveInfo`, `compute_layer_alive_info()`, update `get_sources_by_target()`, update `compute_local_attributions()` |
| `spd/attributions/generate.py` | Add `output_prob_threshold` to config, update `extract_active_components()` |
| `spd/attributions/db.py` | (Optional) Add vocab labels to meta |
| `spd/scripts/local_attributions.html` | Add wte/output colors, update parsing, display token labels |

## Testing Plan

1. **Unit test**: Run `compute_local_attributions()` on a single prompt and verify:
   - `wte` appears as source in some pairs
   - `output` appears as target in some pairs
   - Attribution shapes are correct

2. **Integration test**: Generate a small database (10 prompts) and verify:
   - Database contains wte/output in component_activations table
   - API returns correct data

3. **Visual test**: Load in frontend and verify:
   - wte nodes appear at bottom of graph
   - output nodes appear at top of graph
   - Edges connect properly

## Implementation Order

1. ✅ Understand colleague's implementation (done above)
2. 🔲 Cherry-pick or manually port `compute_layer_alive_info()`
3. 🔲 Update `get_sources_by_target()` with wte/output
4. 🔲 Update `compute_local_attributions()` with wte hook and output handling
5. 🔲 Update `generate.py` config and extraction
6. 🔲 Update frontend colors and parsing
7. 🔲 Test end-to-end

## Code to Copy

The key functions to copy from `feature/global-attr:spd/scripts/calc_local_attributions.py`:

1. Lines 22-54: `LayerAliveInfo` and `compute_layer_alive_info()`
2. Lines 119-138: wte hook setup and cache handling
3. Lines 210-212: wte gradient aggregation (`weighted.sum(dim=1, keepdim=True)`)

From `feature/global-attr:spd/scripts/calc_global_attributions.py`:

1. Lines 196-220: wte hook in `get_sources_by_target()`
2. Lines 223-234: Layer list construction with wte/output
