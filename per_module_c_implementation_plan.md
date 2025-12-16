# Per-Module C Values - Implementation Plan

This document describes the implementation choices for the per-module C values feature.
Use this if you need to continue or modify the implementation.

## Overview

**Goal**: Allow different C values (number of components) for different modules in SPD runs.

**Before**: Single global `C` applied to all modules
**After**: Each pattern specifies its own C value: `[["h.*.mlp.c_fc", 10], ["h.*.attn.*", 20]]`

## Design Decisions

### 1. Field Naming
- **Keep `target_module_patterns`** - do not rename
- **Keep `identity_module_patterns`** - but change its type to also include C

### 2. YAML Syntax
Tuple-like format (list of 2-element lists):
```yaml
target_module_patterns:
  - ["h.*.mlp.c_fc", 10]
  - ["h.*.mlp.down_proj", 20]

identity_module_patterns:
  - ["h.*.attn.o_proj", 15]
```

### 3. Backward Compatibility
Auto-migrate old format with deprecation warning:
```yaml
# Old format (auto-converted with warning)
C: 20
target_module_patterns:
  - "linear1"
  - "linear2"

# Converted to:
target_module_patterns:
  - ["linear1", 20]
  - ["linear2", 20]
```

### 4. Conflict Resolution
When a module matches multiple patterns, **most specific pattern wins**:
- Specificity = number of wildcards (`*`) in pattern
- Fewer wildcards = more specific
- Ties broken alphabetically by pattern name

Example:
```yaml
target_module_patterns:
  - ["h.*.mlp.*", 100]      # 2 wildcards
  - ["h.*.mlp.c_fc", 50]    # 1 wildcard - MORE SPECIFIC
```
Result: `h.0.mlp.c_fc` gets C=50 (more specific wins)

### 5. Identity Patterns
Identity patterns also specify C explicitly (same format as target patterns):
```yaml
identity_module_patterns:
  - ["h.0.attn.o_proj", 30]
```

The `.pre_identity` suffix is added automatically to create `h.0.attn.o_proj.pre_identity`.

### 6. Global C Deprecation
- Remove `C` field from Config (deprecated)
- If old config has `C`, auto-convert to new format with warning
- If new format patterns without global `C`, error with clear message

## Files to Modify

### 1. `spd/utils/module_utils.py`
**Status**: DONE

Added `get_target_module_paths_with_c()` function:
- Input: `list[tuple[str, int]]` - patterns with C values
- Output: `dict[str, int]` - module path to C mapping
- Implements specificity-based conflict resolution

### 2. `spd/configs.py`
**Status**: TODO

Changes needed:
- Change `C` field to `C: PositiveInt | None = None` with deprecation
- Change `target_module_patterns` type to `list[str] | list[tuple[str, int]]`
- Change `identity_module_patterns` type to `list[str] | list[tuple[str, int]] | None`
- Add `model_validator` to auto-convert old format
- Update `all_module_patterns` property to return `list[tuple[str, int]]`

### 3. `spd/models/component_model.py`
**Status**: TODO

Changes needed:
- Change constructor signature: remove `C: int`, change `target_module_patterns` type
- Add `self.module_to_c: dict[str, int]` attribute
- Update `_create_components()` to use per-module C
- Update `_create_ci_fns()` to use per-module C
- Update `from_run_info()` to pass new format

### 4. `spd/run_spd.py`
**Status**: TODO

Changes needed:
- Update `AliveComponentsTracker` initialization to handle per-module C
- No changes to `optimize()` if ComponentModel handles everything

### 5. `spd/metrics/alive_components.py`
**Status**: TODO

Changes needed:
- Constructor should accept `module_to_c: dict[str, int]` instead of single `C`
- Each module's tracker should use its own C value

### 6. Config YAML Files
**Status**: TODO

All configs need to be updated to new format. Key files:
- `spd/experiments/tms/tms_5-2_config.yaml`
- `spd/experiments/tms/tms_5-2-id_config.yaml`
- `spd/experiments/tms/tms_40-10_config.yaml`
- `spd/experiments/tms/tms_40-10-id_config.yaml`
- `spd/experiments/resid_mlp/resid_mlp1_config.yaml`
- `spd/experiments/resid_mlp/resid_mlp2_config.yaml`
- `spd/experiments/resid_mlp/resid_mlp3_config.yaml`
- `spd/experiments/lm/ss_llama_simple_mlp.yaml`
- etc.

### 7. Tests
**Status**: TODO

Test files to update:
- `tests/test_component_model.py` - update all ComponentModel instantiations
- Add new tests for pattern resolution and per-module C

## Implementation Order

1. ✅ `spd/utils/module_utils.py` - Add `get_target_module_paths_with_c()`
2. ✅ `spd/configs.py` - Type changes and validation
3. ✅ `spd/models/component_model.py` - Use per-module C
4. ✅ `spd/metrics/alive_components.py` - Per-module C tracking
5. ✅ `spd/run_spd.py` - Wire everything together
6. ✅ `spd/identity_insertion.py` - Update to accept new format
7. ✅ Tests - Update existing, add new
8. ✅ Config YAML files - Backward compatibility migration added (old format auto-converted with warning)

## Key Code Patterns

### Pattern Resolution (module_utils.py)
```python
def get_target_module_paths_with_c(
    model: nn.Module,
    target_module_patterns: list[tuple[str, int]]
) -> dict[str, int]:
    # Returns {module_path: C_value}
```

### Config Validation (configs.py)
```python
@model_validator(mode="before")
def handle_deprecated_config_keys(cls, config_dict):
    # Auto-convert old format
    if "C" in config_dict and isinstance(config_dict.get("target_module_patterns", [None])[0], str):
        old_c = config_dict.pop("C")
        old_patterns = config_dict["target_module_patterns"]
        config_dict["target_module_patterns"] = [[p, old_c] for p in old_patterns]
        logger.warning("Converted deprecated format...")
```

### ComponentModel Changes
```python
def __init__(self, ..., target_module_patterns: list[tuple[str, int]], ...):
    self.module_to_c = get_target_module_paths_with_c(target_model, target_module_patterns)
    self.target_module_paths = list(self.module_to_c.keys())

    self.components = ComponentModel._create_components(
        target_model=target_model,
        module_to_c=self.module_to_c,  # Pass dict instead of single C
    )
```

### AliveComponentsTracker Changes
```python
def __init__(self, ..., module_to_c: dict[str, int], ...):
    self.n_batches_since_fired: dict[str, Tensor] = {
        m: torch.zeros(c, dtype=torch.int64, device=device)
        for m, c in module_to_c.items()
    }
```

## Testing Strategy

See `per_module_c_test_plan.md` for the full test plan with 30 tests.

Key test categories:
1. Pattern resolution (specificity, conflicts, errors)
2. Config validation (old/new format, migration)
3. ComponentModel integration (per-module C values)
4. Training loop (AliveComponentsTracker, etc.)
5. End-to-end experiments

## Notes for Future Development

1. **Type hints**: Use `list[tuple[str, int]]` for new format, `list[str]` for old
2. **Validation**: Fail fast with clear error messages
3. **Migration**: Log warnings for deprecated format, don't silently break
4. **Testing**: Run `make test` and `make check` after each file change
5. **Configs**: Update all configs in the same PR to avoid confusion
