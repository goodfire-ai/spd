# Global CI Feature Branch Review

**Branch:** `feature/global-ci-fn`
**Date:** 2026-01-19

## What It Implements

The branch adds a **Global CI Function** architecture that computes causal importance scores for ALL layers simultaneously using a single shared MLP, as opposed to the existing **Layerwise CI** approach with independent CI functions per layer.

**Key architectural difference:**
```
Layerwise:  Layer1 Input → CI_Fn_1 → CI₁
            Layer2 Input → CI_Fn_2 → CI₂

Global:     Layer1 Input ┐
            Layer2 Input ├→ Concat → GlobalMLP → Split → CI₁, CI₂
            Layer3 Input ┘
```

**Key files:**
- `spd/configs.py:19-43` - Discriminated union `CiConfig = LayerwiseCiConfig | GlobalCiConfig`
- `spd/models/components.py:113-171` - `GlobalSharedMLPCiFn` class
- `spd/models/component_model.py:104-293` - CI function creation and routing
- `spd/spd_types.py:53-54` - Type definitions for CI function types

---

## Issues Identified

### Critical (Must Fix Before Merge)

#### Issue 1: Config Migration Missing ✅ FIXED
**Status:** Fixed by merging `feature/global-ci-fn-config-migration` branch

Existing YAML configs use old flat format (`ci_fn_type`, `ci_fn_hidden_dims`) but the new code expects nested `ci_config` structure. This will break all existing experiments.

---

#### Issue 2: Incorrect Assertion Logic
**File:** `spd/models/component_model.py:580,584`

```python
# Current (wrong):
assert lower_leaky_output.all() <= 1.0
# Should be:
assert (lower_leaky_output <= 1.0).all()
```

**Status:** [ ] TODO

---

#### Issue 3: No Test Coverage
**Status:** Deferred - will address after issues 4-8

All 21 tests in `test_component_model.py` use `LayerwiseCiConfig` only. No tests exist for global CI path.

---

### Important (Should Fix)

#### Issue 4: Non-deterministic Layer Ordering (CRITICAL)
**File:** `spd/models/components.py:124`

```python
self.layer_order = list(layer_configs.keys())
```

Depends on dict iteration order. Should explicitly sort for reproducibility.

**Status:** [ ] TODO

---

#### Issue 5: Missing Config Validation
No validation that `fn_type` is compatible with `mode` (e.g., using `global_shared_mlp` with `mode: "layerwise"` would pass Pydantic but fail at runtime).

**Status:** [ ] TODO

---

#### Issue 6: Code Duplication
**File:** `spd/models/component_model.py`

Input dimension extraction logic duplicated between `_create_ci_fn` (lines 220-228) and `_create_global_ci_fn` (lines 270-285).

**Status:** [ ] TODO

---

#### Issue 7: Checkpoint Compatibility (IMPORTANT - fail-fast)
Loading a layerwise checkpoint with global config (or vice versa) will fail silently.

**Status:** [ ] TODO

---

### Minor

#### Issue 8: Type Annotation
**File:** `spd/models/component_model.py:128`

`self.global_ci_fn: nn.Module | None` could be more specific as `GlobalSharedMLPCiFn | None`

**Status:** [ ] TODO

---

## What's Working Well

- **API Consistency**: Both paths return identical `CIOutputs` structure - callers don't need to know which mode is active
- **Discriminated Union Pattern**: Clean separation via `mode: "layerwise" | "global"`
- **Extensibility**: Adding new global CI types is straightforward (add to types, implement class, add to factory)
- **Sigmoid Logic**: `_apply_sigmoid_to_ci_outputs()` is correctly shared between both paths
