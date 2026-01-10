# Refactor Plan: Centralize Component Activation Computation

## Goal
Eliminate redundant `get_component_acts` calls by computing component activations once and passing them through.

## Current Problem
- `extract_node_subcomp_acts` in `spd/app/backend/compute.py` calls `model.components[layer].get_component_acts(input_acts)` for each layer
- This duplicates computation already done (or easily done) elsewhere
- The function also takes `model` as a parameter, coupling it to ComponentModel

## Solution
Add `get_all_component_acts` method to `ComponentModel` and refactor callers to compute once and pass through.

---

## Step 1: Add method to ComponentModel

**File:** `spd/models/component_model.py`

**Location:** After `calc_causal_importances` method (around line 543)

**Code to add:**
```python
def get_all_component_acts(
    self,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Compute component activations (v_i^T @ a) for all layers.

    Args:
        pre_weight_acts: Dict mapping layer name to input activations.

    Returns:
        Dict mapping layer name to component activations tensor.
    """
    return {
        layer: self.components[layer].get_component_acts(acts)
        for layer, acts in pre_weight_acts.items()
        if layer in self.components
    }
```

---

## Step 2: Update extract_node_subcomp_acts signature

**File:** `spd/app/backend/compute.py`

**Change function signature from:**
```python
def extract_node_subcomp_acts(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "1 seq d_in"]],
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]] | None = None,
    ci_threshold: float = 0.0,
) -> dict[str, float]:
```

**To:**
```python
def extract_node_subcomp_acts(
    component_acts: dict[str, Float[Tensor, "1 seq C"]],
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]] | None = None,
    ci_threshold: float = 0.0,
) -> dict[str, float]:
```

**Update function body:**
- Remove the `if layer_name not in model.components: continue` check (no longer needed)
- Remove `subcomp_acts = model.components[layer_name].get_component_acts(input_acts)`
- Iterate directly over `component_acts.items()` instead of `pre_weight_acts.items()`
- Use `subcomp_acts` directly from the dict value

---

## Step 3: Update compute_edges_from_ci

**File:** `spd/app/backend/compute.py`

**Location:** Around line 377-380

**Change from:**
```python
node_subcomp_acts = extract_node_subcomp_acts(
    model, pre_weight_acts, ci_lower_leaky=ci_lower_leaky, ci_threshold=0.0
)
```

**To:**
```python
component_acts = model.get_all_component_acts(pre_weight_acts)
node_subcomp_acts = extract_node_subcomp_acts(
    component_acts, ci_lower_leaky=ci_lower_leaky, ci_threshold=0.0
)
```

---

## Step 4: Update CIOnlyResult and compute_ci_only

**File:** `spd/app/backend/compute.py`

**Update CIOnlyResult dataclass (around line 492):**
```python
@dataclass
class CIOnlyResult:
    """Result of computing CI values only (no attribution graph)."""
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq n_components"]]
    output_probs: Float[Tensor, "1 seq vocab"]
    pre_weight_acts: dict[str, Float[Tensor, "1 seq d_in"]]
    component_acts: dict[str, Float[Tensor, "1 seq C"]]  # ADD THIS
```

**Update compute_ci_only function (around line 516-529):**
```python
def compute_ci_only(...) -> CIOnlyResult:
    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        ci = model.calc_causal_importances(...)
        output_probs = torch.softmax(output_with_cache.output, dim=-1)
        component_acts = model.get_all_component_acts(output_with_cache.cache)  # ADD

    return CIOnlyResult(
        ci_lower_leaky=ci.lower_leaky,
        output_probs=output_probs,
        pre_weight_acts=output_with_cache.cache,
        component_acts=component_acts,  # ADD
    )
```

---

## Step 5: Update probe endpoint

**File:** `spd/app/backend/routers/activation_contexts.py`

**Location:** Around line 144-147

**Change from:**
```python
input_acts = result.pre_weight_acts[request.layer]
subcomp_acts_tensor = loaded.model.components[request.layer].get_component_acts(input_acts)
subcomp_acts = subcomp_acts_tensor[0, :, request.component_idx].tolist()
```

**To:**
```python
subcomp_acts_tensor = result.component_acts[request.layer]
subcomp_acts = subcomp_acts_tensor[0, :, request.component_idx].tolist()
```

---

## Step 6: Update harvest module

**File:** `spd/harvest/harvest.py`

**Location:** Around lines 197-206 and 285-292 (two occurrences)

**Change from:**
```python
subcomp_acts: Float[Tensor, "B S n_comp"] = torch.cat(
    [
        model.components[layer].get_component_acts(out.cache[layer])
        for layer in layer_names
    ],
    dim=2,
)
```

**To:**
```python
component_acts = model.get_all_component_acts(out.cache)
subcomp_acts: Float[Tensor, "B S n_comp"] = torch.cat(
    [component_acts[layer] for layer in layer_names],
    dim=2,
)
```

---

## Step 7: Run checks

```bash
make check  # Python type checking
cd spd/app/frontend && npm run check  # Frontend (unchanged but verify)
make test  # Run tests
```

---

## Files Modified Summary

1. `spd/models/component_model.py` - Add `get_all_component_acts` method
2. `spd/app/backend/compute.py` - Update `extract_node_subcomp_acts`, `CIOnlyResult`, `compute_ci_only`, `compute_edges_from_ci`
3. `spd/app/backend/routers/activation_contexts.py` - Update probe endpoint
4. `spd/harvest/harvest.py` - Update both harvest functions

## Expected Benefits

- Single point of component activation computation
- `extract_node_subcomp_acts` becomes a pure dict transformation (no model dependency)
- Cleaner separation of concerns
- Harvest module uses same pattern as app
