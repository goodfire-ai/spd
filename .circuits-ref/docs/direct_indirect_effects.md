# Measuring Direct vs Indirect Neuron Effects

This document proposes a method for decomposing a neuron's total effect on logits into **direct** and **indirect** components, and explains why alternative approaches are insufficient.

## Problem Statement

When a neuron fires in an MLP layer, it adds a vector V to the residual stream:
```
residual_out = residual_in + V
```

This V affects the final logits through two pathways:

1. **Direct effect**: V passes through the residual stream and is projected to logits via `lm_head`
2. **Indirect effect**: V triggers downstream neurons, which contribute their own vectors

We want to quantify: "How much of this neuron's effect on logits comes from its own contribution vs triggering downstream circuits?"

## Recommended Method: Activation Freezing

### Approach

Use causal intervention with frozen downstream activations:

1. **Clean pass**: Run the model normally, cache all layer outputs
2. **Direct pass**: Inject V at layer L, but **freeze** all downstream layers (L+1 to 31) to their clean outputs plus V (V passes through residual, but downstream neurons don't react)
3. **Total pass**: Inject V at layer L, let downstream layers react naturally

```python
Direct effect  = logits(inject V, freeze downstream) - logits(clean)
Total effect   = logits(inject V, normal forward) - logits(clean)
Indirect effect = Total effect - Direct effect
```

### Implementation

```python
def decompose_with_freezing(model, inputs, layer, neuron, scale=1.0):
    """
    Decompose neuron effect into direct and indirect components.

    Returns:
        direct_effect: Logit change from V passing through residual only
        indirect_effect: Additional logit change from downstream neurons reacting
        total_effect: Full logit change when neuron fires
    """
    down_proj = model.model.layers[layer].mlp.down_proj.weight.data
    V = down_proj[:, neuron].clone() * scale

    # Step 1: Clean pass - cache all layer outputs
    clean_cache = {}

    def make_cache_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            clean_cache[layer_idx] = hidden.detach().clone()
        return hook

    handles = [model.model.layers[i].register_forward_hook(make_cache_hook(i))
               for i in range(32)]

    with torch.no_grad():
        clean_out = model(**inputs)
        clean_logits = clean_out.logits[:, -1, :].clone()

    for h in handles:
        h.remove()

    # Step 2: Direct pass - inject V, freeze downstream
    def inject_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += V
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def make_freeze_hook(layer_idx):
        def hook(module, input, output):
            # Replace with clean output, but V passes through residual
            frozen = clean_cache[layer_idx].clone()
            frozen[:, -1, :] += V
            return (frozen,) + output[1:] if isinstance(output, tuple) else frozen
        return hook

    handles = [model.model.layers[layer].register_forward_hook(inject_hook)]
    handles += [model.model.layers[i].register_forward_hook(make_freeze_hook(i))
                for i in range(layer + 1, 32)]

    with torch.no_grad():
        direct_out = model(**inputs)
        direct_logits = direct_out.logits[:, -1, :].clone()

    for h in handles:
        h.remove()

    # Step 3: Total pass - inject V, normal forward
    h = model.model.layers[layer].register_forward_hook(inject_hook)

    with torch.no_grad():
        total_out = model(**inputs)
        total_logits = total_out.logits[:, -1, :].clone()

    h.remove()

    # Compute effects
    direct_effect = direct_logits - clean_logits
    total_effect = total_logits - clean_logits
    indirect_effect = total_effect - direct_effect

    return direct_effect, indirect_effect, total_effect
```

### Results

Empirical measurements on Llama-3.1-8B-Instruct:

| Layer | Direct Mag | Indirect Mag | Total Mag | Direct/Total |
|-------|------------|--------------|-----------|--------------|
| L5    | ~78        | ~482         | ~461      | 17%          |
| L10   | ~81        | ~190         | ~174      | 46%          |
| L15   | ~81        | ~299         | ~290      | 28%          |
| L20   | ~82        | ~145         | ~159      | 52%          |
| L25   | ~86        | ~109         | ~139      | 62%          |
| L30   | ~82        | ~53          | ~90       | 91%          |

**Key finding**: Direct effect magnitude is nearly constant (~80-86) across layers because V passes through the residual unchanged. What varies is the indirect effect—how much downstream neurons amplify or transform the signal.

Note: Direct + Indirect ≠ Total because the effects can interfere constructively or destructively (they're not orthogonal in logit space).

### Interpretation for Labeling

| Direct/Total Ratio | Classification | Interpretation |
|--------------------|----------------|----------------|
| > 0.7              | Logit-dominant | Neuron's own direction matters most |
| 0.3 - 0.7          | Mixed          | Both direct output and downstream triggering |
| < 0.3              | Routing-dominant | Effect is primarily through triggering downstream |

### Computational Cost

- 3 forward passes per neuron
- For 8K neurons: ~24K forward passes
- Can be batched by injecting multiple neurons at different positions

---

## Alternative Methods Considered

### Method 1: Output Projection Comparison

**Approach**: Compare `lm_head @ V` (direct logit contribution) to RelP edge weights (downstream influence).

```python
direct_effect = lm_head @ down_proj[:, neuron]  # [vocab]
downstream_effect = max(relp_edge_weights_from_neuron)
ratio = direct_effect.abs().max() / downstream_effect
```

**Why rejected**:
- The two quantities are not comparable (different units/scales)
- RelP edge weights measure gradient-based attribution, not causal effect
- No principled way to normalize for comparison

### Method 2: Projection onto V's Logit Direction

**Approach**: Decompose total logit delta into V-aligned and orthogonal components.

```python
total_delta = logits(with_neuron) - logits(without_neuron)
V_logit_dir = normalize(lm_head @ V)
direct = |total_delta · V_logit_dir|
indirect = |total_delta - direct * V_logit_dir|
```

**Why rejected**:
- Indirect effects can also be aligned with V's direction
- If neuron triggers downstream neurons promoting the same tokens, that gets incorrectly counted as "direct"
- Conflates "V-aligned" with "direct causation"

### Method 3: Late-Layer Injection Comparison

**Approach**: Compare effect of injecting V at layer L vs injecting at layer 31.

```python
total_effect = logits(inject at L) - logits(clean)
direct_effect = logits(inject at L31) - logits(clean)
indirect = total - direct
```

**Why rejected**:
- Injecting at L31 doesn't account for how V would have been processed by final LayerNorm
- The residual stream context at L31 is different from what it would be if V had propagated naturally
- Overestimates direct effect because V doesn't experience any transformation

### Method 4: Perturbation Direction Analysis

**Approach**: Track how the perturbation delta rotates through layers.

```python
# Inject V at layer L, measure delta at each subsequent layer
delta_L = hidden_inject[L] - hidden_clean[L]  # = V
delta_31 = hidden_inject[31] - hidden_clean[31]
rotation = cosine_similarity(delta_L, delta_31)
```

**Findings**: This analysis was useful for understanding the mechanism:
- Early layers (L5): cos_sim ≈ 0.02 (signal completely transformed)
- Late layers (L30): cos_sim ≈ 0.85 (signal mostly preserved)

**Why insufficient for decomposition**:
- Tells us about direction change, not magnitude of direct vs indirect effects
- A rotated signal could still be "direct" if downstream neurons just transform V without adding new information
- Doesn't give us the causal decomposition needed for labeling

### Method 5: Gradient-Based Attribution

**Approach**: Use gradients to attribute logit changes to direct vs indirect pathways.

```python
# Compute gradient of logit w.r.t. neuron activation
# Decompose into path through residual vs path through downstream neurons
```

**Why rejected**:
- Gradients measure local sensitivity, not causal effect
- Path decomposition in transformers is intractable (exponential paths)
- RelP already does gradient-based attribution; we want something complementary

---

## Why Freezing Works

The freezing method provides a clean causal interpretation:

1. **Counterfactual reasoning**: "What would happen if downstream neurons couldn't react to V?"
   - Direct effect answers this precisely

2. **Residual stream semantics**: V passes through the residual unchanged in both conditions
   - The only difference is whether downstream neurons process V or not

3. **Additivity**: Direct + Indirect ≈ Total (with some interference)
   - This decomposition is meaningful because the mechanisms are separable

4. **Layer-agnostic**: Same method works at any layer
   - Results are comparable across the network

---

## Limitations

1. **Computational cost**: 3 forward passes per neuron is expensive for full-network analysis

2. **Context dependence**: The ratio depends on the input prompt
   - A neuron might be routing-dominant for one prompt and logit-dominant for another

3. **Interference effects**: Direct and indirect effects can interfere
   - Direct + Indirect ≠ Total in general
   - The "indirect" component includes both amplification AND cancellation

4. **Single-neuron assumption**: Measures effect of one neuron in isolation
   - Doesn't capture interactions between multiple neurons firing together

---

## Recommendations

1. **For labeling**: Compute direct/total ratio for each neuron on representative prompts
   - Use median across prompts if context-dependence is high

2. **For efficiency**: Sample neurons per layer rather than computing for all
   - Layer-level statistics may be sufficient for many analyses

3. **For interpretation**: Combine with output projection analysis
   - High direct ratio + strong logit projection → "logit-shaping neuron"
   - Low direct ratio + weak logit projection → "routing/triggering neuron"
