# DatasetAttributionStorage vs ModelAdapter: Topology Duplication

## Problem

`DatasetAttributionStorage` and `ModelAdapter` both encode model topology — the
set of layers, what counts as a source/target, and the special "wte"/"output"
pseudo-layers. This creates duplicated concepts that must stay in sync.

### What DatasetAttributionStorage knows about topology:
- Key format: "wte:{token_id}", "layer:c_idx", "output:{token_id}"
- Source/target validity: wte and components are sources; components and output are targets
- Component ordering: `component_layer_keys` list with index mapping
- Vocab size (for wte/output index ranges)
- Output attribution requires `w_unembed` (which it gets from ModelAdapter)

### What ModelAdapter knows about topology:
- `target_module_paths`: ordered list of component layers
- `embedding_path` / `unembed_path`: actual module paths
- `ordered_layers()`: ["wte", ...components..., "output"]
- `get_unembed_weight()`: the w_unembed tensor
- Cross-sequence pair detection (kv/o roles)

### The overlap:
- Both maintain an ordered list of component layers
- Both understand "wte" and "output" as pseudo-layers
- Both know which keys can be sources vs targets
- `DatasetAttributionStorage._parse_key()` duplicates the "layer:idx" convention
  used throughout the app

## Consequences

1. If a new pseudo-layer is added, both need updating
2. The key format ("layer:c_idx") is hardcoded in storage but is actually a
   codebase-wide convention — no single owner
3. `DatasetAttributionStorage` query methods need `w_unembed` passed in from
   the outside (from `ModelAdapter.get_unembed_weight()`) — the storage class
   depends on the adapter but doesn't know about it

## Possible directions

**A. Storage stores raw tensors, adapter owns indexing.**
`DatasetAttributionStorage` becomes a dumb tensor container (matrices + metadata).
A query layer uses `ModelAdapter` to resolve keys → indices. This separates
"what was harvested" from "how to query it."

**B. Shared indexing module.**
Extract the key format + source/target validity + index mapping into a small
shared module that both `ModelAdapter` and `DatasetAttributionStorage` use.
Neither reinvents the convention.

**C. Leave it.**
The duplication is internally consistent and only matters if topology changes.
The cost of refactoring may not be worth it for a research codebase.

## Recommendation

Lean toward B if we find ourselves editing both classes when topology changes.
Otherwise C is fine — the current code works, it's just not beautiful.
