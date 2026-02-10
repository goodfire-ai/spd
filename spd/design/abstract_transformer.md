# Abstract Transformer Model

## Problem

Transformer model topology is encoded in 6+ places, each with its own path parsing, role detection, and architectural assumptions:

| Consumer | What it knows | How it gets there |
|---|---|---|
| `ModelAdapter` / `ArchConfig` | embedding, unembed, kv/o roles, role groups | `match` on model class, glob patterns |
| `DatasetAttributionStorage` | source/target validity, key format | Hardcoded "wte:"/"output:" prefix parsing |
| `compact_skeptical.py` | layer descriptions for LLM prompts | `split(".")`, hardcoded sublayer dict |
| `compute.py` | "wte"/"output" pseudo-layers, block count | String comparison, `match` on model class |
| `harvester.py` / `harvest.py` | layer iteration, embedding/unembed modules | `model.target_module_paths`, adapter |
| `ComponentModel` | module paths, component counts | `module_to_c` dict from config patterns |

Every new model architecture requires changes in multiple places. Every consumer has its own way of extracting the same information from dotted path strings.

## Design

### Core Abstraction

A `TransformerTopology` that maps concrete module paths onto an abstract structure:

```python
@dataclass(frozen=True)
class TransformerTopology:
    """Abstract transformer topology resolved at model load time.

    All consumers use this instead of parsing paths or matching model classes.
    """

    embedding: LayerInfo          # wte / transformer.wte
    blocks: list[BlockInfo]       # ordered transformer blocks
    unembed: LayerInfo            # lm_head

    # Derived lookups (built in __post_init__)
    path_to_abstract: dict[str, AbstractRef]   # "h.0.attn.q_proj" -> blocks[0].attn.q
    abstract_to_path: dict[AbstractRef, str]   # blocks[0].attn.q -> "h.0.attn.q_proj"


@dataclass(frozen=True)
class LayerInfo:
    path: str                     # concrete module path
    module: nn.Module             # resolved module reference


@dataclass(frozen=True)
class BlockInfo:
    index: int                    # 0-based block index
    attn: AttentionInfo | None    # None if MLP-only model
    ffn: FFNInfo


@dataclass(frozen=True)
class AttentionInfo:
    """Attention projections in this block.

    Handles both separate (q, k, v, o) and fused (qkv) projections.
    """
    q: LayerInfo | None           # None if fused
    k: LayerInfo | None           # None if fused
    v: LayerInfo | None           # None if fused
    qkv: LayerInfo | None         # None if separate
    o: LayerInfo


@dataclass(frozen=True)
class FFNInfo:
    """FFN projections in this block.

    Handles both standard (up, down) and SwiGLU (gate, up, down).
    """
    up: LayerInfo
    down: LayerInfo
    gate: LayerInfo | None        # None if not SwiGLU
```

### Construction

One `match` on model class, one place:

```python
def build_topology(model: ComponentModel) -> TransformerTopology:
    """Build topology by matching the model's architecture."""
    target_model = model.target_model
    target_paths = model.target_module_paths

    arch = _get_arch_config(target_model)  # existing ArchConfig match
    # ... resolve all modules, build BlockInfo list
```

### What Consumers Get

**ModelAdapter** becomes a thin wrapper or merges into topology:
```python
# Cross-seq detection
topology.is_cross_seq_pair(source_path, target_path)
# -> True if source is k/v and target is o in same block

# Ordered layers for gradient computation
topology.ordered_paths()
# -> ["wte", "h.0.mlp.c_fc", "h.0.attn.q_proj", ..., "output"]
```

**Autointerp prompt formatting**:
```python
ref = topology.path_to_abstract["h.1.mlp.down_proj"]
# ref.block_index = 1, ref.role = "ffn.down"
ref.describe(n_blocks=12)
# -> "MLP down-projection in layer 2 of 12"
```

**DatasetAttributionStorage** key resolution:
```python
topology.source_index("h.0.attn.q_proj:5")  # -> vocab_size + component_offset
topology.is_source("wte:123")               # -> True
topology.is_target("output:456")            # -> True
```

**Block counting**:
```python
len(topology.blocks)  # no more match-on-model-class
```

### What Changes

| Before | After |
|---|---|
| `_extract_block_index(path)` | `topology.path_to_abstract[path].block_index` |
| `_extract_role(path)` | `topology.path_to_abstract[path].role` |
| `_parse_layer_description(layer, n)` | `topology.path_to_abstract[layer].describe(n)` |
| `model.target_model.wte` | `topology.embedding.module` |
| `adapter.unembed_module` | `topology.unembed.module` |
| `adapter.is_cross_seq_pair(s, t)` | `topology.is_cross_seq_pair(s, t)` |
| `get_model_n_blocks(model)` | `len(topology.blocks)` |
| `"wte" literal checks` | `path == topology.embedding.path` or typed enum |

### What Doesn't Change

- `ComponentModel` still owns `module_to_c` and component creation
- Config still uses fnmatch patterns for target module selection
- The discriminated union of CI functions stays as-is
- Loss functions don't care about topology

### Migration Path

1. Build `TransformerTopology` alongside existing `ModelAdapter` (topology constructed from same `ArchConfig`)
2. Migrate consumers one at a time (harvest, attributions, autointerp, app compute)
3. Once all consumers use topology, `ModelAdapter` either becomes a thin facade or merges in
4. Remove scattered path parsing utilities

### Open Questions

- Should `TransformerTopology` own the component count per layer (`C`), or does that stay on `ComponentModel`? Topology is about model structure; component count is about the decomposition.
- Should the abstract refs use string keys ("ffn.down") or an enum? Strings are more extensible; enums catch typos.
- How to handle models where a role appears in both attn and MLP (GPT2's `c_proj`)? The glob patterns in `ArchConfig` already handle this — topology should preserve the disambiguation.
