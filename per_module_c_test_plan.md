# Per-Module C Values - Test Plan

This file tracks the acceptance tests for the per-module C values feature.
Only update `passes: true` after explicitly running and verifying each test.

## Feature Requirements

1. Deprecate global `C` - all patterns must specify their C values
2. If a module matches multiple patterns, most specific wins (fewest wildcards)
3. All patterns must specify C (no mixing)
4. YAML syntax: `["h.*.mlp.c_fc", 10]` (tuple-like format)
5. Keep field name `target_module_patterns` (not renaming)
6. Identity patterns also specify C explicitly: `identity_module_patterns: [["pattern", C], ...]`
7. Auto-migrate old format with deprecation warning

---

## Unit Tests: Pattern Resolution

### Test 1: Basic pattern matching with C values
- **Description**: Patterns without wildcards should match exact module names and return correct C
- **Input**: `target_modules = [["linear1", 10], ["linear2", 20]]`
- **Expected**: `{"linear1": 10, "linear2": 20}`
- **passes**: false

### Test 2: Wildcard pattern expansion
- **Description**: Pattern `h.*.mlp.c_fc` should expand to all matching modules with same C
- **Input**: Model with `h.0.mlp.c_fc`, `h.1.mlp.c_fc`; pattern `[["h.*.mlp.c_fc", 50]]`
- **Expected**: `{"h.0.mlp.c_fc": 50, "h.1.mlp.c_fc": 50}`
- **passes**: false

### Test 3: Specificity resolution - fewer wildcards wins
- **Description**: When module matches multiple patterns, pattern with fewer wildcards wins
- **Input**: `[["h.*.mlp.*", 100], ["h.*.mlp.c_fc", 50]]` for module `h.0.mlp.c_fc`
- **Expected**: `h.0.mlp.c_fc` gets C=50 (1 wildcard beats 2 wildcards)
- **passes**: false

### Test 4: Specificity resolution - exact match wins
- **Description**: Exact pattern (0 wildcards) beats any wildcard pattern
- **Input**: `[["linear*", 100], ["linear1", 10]]`
- **Expected**: `linear1` gets C=10, `linear2` gets C=100
- **passes**: false

### Test 5: Equal specificity conflict - error raised
- **Description**: If two patterns with same wildcard count match same module, raise error
- **Input**: `[["h.0.*", 10], ["*.mlp", 20]]` for module `h.0.mlp`
- **Expected**: `ValueError` with clear message about conflict
- **passes**: false

### Test 6: Unmatched pattern - error raised
- **Description**: Pattern that matches no modules should raise error
- **Input**: `[["nonexistent_module", 10]]`
- **Expected**: `ValueError` with message "did not match any modules"
- **passes**: false

### Test 7: Empty pattern list - error raised
- **Description**: Empty target_modules should raise validation error
- **Input**: `target_modules = []`
- **Expected**: `ValueError` about empty patterns
- **passes**: false

---

## Unit Tests: Config Validation

### Test 8: Old format with global C raises deprecation error
- **Description**: Using old `C: 20` + `target_module_patterns: ["linear1"]` should raise error
- **Input**: Config with both `C` and `target_module_patterns` as list of strings
- **Expected**: `ValueError` explaining new format required
- **passes**: false

### Test 9: New tuple format is accepted
- **Description**: New format `target_modules: [["linear1", 20]]` should parse correctly
- **Input**: YAML with tuple format
- **Expected**: Config loads successfully with correct values
- **passes**: false

### Test 10: Invalid tuple format - wrong length
- **Description**: Tuple with wrong number of elements should error
- **Input**: `target_modules: [["linear1", 20, "extra"]]`
- **Expected**: `ValueError` about tuple format
- **passes**: false

### Test 11: Invalid tuple format - wrong types
- **Description**: Non-string pattern or non-int C should error
- **Input**: `target_modules: [[123, "not_an_int"]]`
- **Expected**: `ValueError` about types
- **passes**: false

### Test 12: Negative or zero C value - error raised
- **Description**: C must be positive integer
- **Input**: `target_modules: [["linear1", 0]]` or `[["linear1", -5]]`
- **Expected**: `ValueError` about positive integer
- **passes**: false

---

## Integration Tests: ComponentModel

### Test 13: ComponentModel creates components with per-module C
- **Description**: Each module should get its own C value from pattern
- **Input**: ComponentModel with `[["linear1", 10], ["linear2", 20]]`
- **Expected**: `components["linear1"].C == 10`, `components["linear2"].C == 20`
- **passes**: false

### Test 14: ComponentModel creates ci_fns with per-module C
- **Description**: CI functions should also use per-module C
- **Input**: ComponentModel with `[["linear1", 10], ["linear2", 20]]`
- **Expected**: ci_fns have correct output dimensions matching C
- **passes**: false

### Test 15: ComponentModel.module_to_c dict is populated
- **Description**: ComponentModel should expose module_to_c mapping
- **Input**: ComponentModel with multiple patterns
- **Expected**: `model.module_to_c` returns correct dict
- **passes**: false

### Test 16: ComponentModel forward pass works with different C values
- **Description**: Forward pass should work when modules have different C
- **Input**: ComponentModel with varying C, run forward pass
- **Expected**: No errors, output shape correct
- **passes**: false

### Test 17: ComponentModel weight_deltas work with different C values
- **Description**: calc_weight_deltas should work with per-module C
- **Input**: ComponentModel with varying C
- **Expected**: weight_deltas dict has correct shapes per module
- **passes**: false

### Test 18: ComponentModel causal_importances work with different C values
- **Description**: calc_causal_importances should return correct shapes
- **Input**: ComponentModel with varying C, compute CIs
- **Expected**: CI tensors have shape `(..., C)` where C varies per module
- **passes**: false

---

## Integration Tests: Full Training Loop

### Test 19: run_spd.optimize works with per-module C
- **Description**: Full optimization loop should run without errors
- **Input**: Config with per-module C, run for a few steps
- **Expected**: No errors, loss decreases
- **passes**: false

### Test 20: AliveComponentsTracker works with per-module C
- **Description**: Tracker should handle different C per module
- **Input**: AliveComponentsTracker with module_to_c dict
- **Expected**: Tracks alive components correctly per module
- **passes**: false

### Test 21: Checkpoint save/load works with per-module C
- **Description**: Model can be saved and loaded with different C values
- **Input**: Train model, save, load from checkpoint
- **Expected**: Loaded model has same C values per module
- **passes**: false

---

## Integration Tests: Identity Patterns

### Test 22: Identity patterns specify C explicitly
- **Description**: Identity patterns now use tuple format with C values
- **Input**: `target_module_patterns: [["h.*.mlp", 50]]`, `identity_module_patterns: [["h.0.attn", 30]]`
- **Expected**: `h.0.attn.pre_identity` gets C=30
- **passes**: false

### Test 23: Old identity format auto-migrates with global C
- **Description**: Old string format with global C auto-converts with warning
- **Input**: Old config with `C: 20`, `identity_module_patterns: ["h.0.attn"]`
- **Expected**: Auto-converts to `[["h.0.attn", 20]]` with deprecation warning
- **passes**: false

### Test 24: Identity patterns without C and no global C - error
- **Description**: If old format used without global C, raise clear error
- **Input**: New `target_module_patterns: [[...]]` with old `identity_module_patterns: [...]`
- **Expected**: `ValueError` explaining new format required
- **passes**: false

---

## End-to-End Tests

### Test 25: TMS experiment runs with per-module C
- **Description**: TMS 5-2 experiment should run with new config format
- **Input**: TMS config with `target_modules: [["linear1", 20], ["linear2", 20]]`
- **Expected**: Experiment runs, produces expected outputs
- **passes**: false

### Test 26: TMS experiment with different C per layer
- **Description**: TMS should work with different C for each layer
- **Input**: TMS config with `target_modules: [["linear1", 10], ["linear2", 30]]`
- **Expected**: Experiment runs, components have different sizes
- **passes**: false

### Test 27: LM experiment runs with per-module C (if applicable)
- **Description**: Language model experiment should run with new format
- **Input**: LM config with per-module C values
- **Expected**: Experiment runs without errors
- **passes**: false

---

## Regression Tests

### Test 28: Existing tests still pass after changes
- **Description**: All existing tests in test suite should pass
- **Command**: `make test`
- **Expected**: All tests pass
- **passes**: false

### Test 29: Type checking passes
- **Description**: Pyright should report no new errors
- **Command**: `make type`
- **Expected**: No type errors
- **passes**: false

### Test 30: Linting passes
- **Description**: Ruff linting should pass
- **Command**: `make check`
- **Expected**: No lint errors
- **passes**: false

---

## Summary

| Category | Total | Passed | Failed |
|----------|-------|--------|--------|
| Pattern Resolution | 7 | 0 | 7 |
| Config Validation | 5 | 0 | 5 |
| ComponentModel Integration | 6 | 0 | 6 |
| Full Training Loop | 3 | 0 | 3 |
| Identity Patterns | 3 | 0 | 3 |
| End-to-End | 3 | 0 | 3 |
| Regression | 3 | 0 | 3 |
| **TOTAL** | **30** | **0** | **30** |

**Feature Status**: NOT READY - 0/30 tests passing
