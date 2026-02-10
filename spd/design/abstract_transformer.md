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

Every new model architecture requires changes in multiple places.

## Design

### Core Types

```python
# Roles as literal unions — predictable, exhaustive matching
AttentionRole = Literal["q", "k", "v", "qkv", "o"]
FFNRole = Literal["up", "down", "gate"]

@dataclass(frozen=True)
class LayerInfo:
    path: str                     # concrete module path ("h.0.attn.q_proj")
    module: nn.Module             # resolved module reference
    role: AttentionRole | FFNRole # abstract role

# Discriminated union for attention variants — no parallel optionals
@dataclass(frozen=True)
class SeparateAttention:
    q: LayerInfo
    k: LayerInfo
    v: LayerInfo
    o: LayerInfo

@dataclass(frozen=True)
class FusedAttention:
    qkv: LayerInfo
    o: LayerInfo

AttentionInfo = SeparateAttention | FusedAttention

# FFN variants
@dataclass(frozen=True)
class StandardFFN:
    up: LayerInfo
    down: LayerInfo

@dataclass(frozen=True)
class SwiGLUFFN:
    gate: LayerInfo
    up: LayerInfo
    down: LayerInfo

FFNInfo = StandardFFN | SwiGLUFFN

@dataclass(frozen=True)
class BlockInfo:
    index: int                    # 0-based block index
    attn: AttentionInfo | None    # None if MLP-only model
    ffn: FFNInfo
```

### TransformerTopology

```python
class TransformerTopology:
    """Abstract transformer topology resolved at model load time.

    Single source of truth for model structure. All consumers use this
    instead of parsing paths or matching model classes.
    """

    def __init__(self, model: ComponentModel):
        target_model = model.target_model
        arch = _get_arch_config(target_model)  # existing match-on-class

        # Resolve embedding + unembed
        self.embedding = LayerInfo(...)
        self.unembed = LayerInfo(...)

        # Build blocks from target_module_paths
        self.blocks: list[BlockInfo] = _build_blocks(...)

        # Derived lookups
        self._path_to_block: dict[str, tuple[int, AttentionRole | FFNRole]] = ...

    # --- Queries ---

    def block_index(self, path: str) -> int:
        """Block index for a concrete module path."""

    def role(self, path: str) -> AttentionRole | FFNRole:
        """Abstract role for a concrete module path."""

    def describe(self, path: str) -> str:
        """Human-readable description for prompts."""

    def is_cross_seq_pair(self, source: str, target: str) -> bool:
        """True if source is k/v and target is o in same block."""

    def ordered_paths(self) -> list[str]:
        """All paths in execution order: [embedding, ...components..., unembed]."""

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)
```

### What Consumers Get

**ModelAdapter** merges into / wraps topology. Role groups and display names move here.

**Autointerp**:
```python
topology.describe("h.1.mlp.down_proj")
# -> "MLP down-projection in layer 2 of 12"
```

**Attributions**: `topology.embedding.module` and `topology.unembed.module` instead of hardcoded paths.

**Block counting**: `topology.n_blocks` instead of `match` on model class.

### What Doesn't Change

- `ComponentModel` still owns `module_to_c` and component creation (C is about decomposition, not topology)
- Config still uses fnmatch patterns for target module selection
- CI functions stay as-is
- Loss functions don't care about topology

### Migration Path

1. Build `TransformerTopology` alongside existing `ModelAdapter`
2. Migrate consumers one at a time
3. `ModelAdapter` becomes a thin facade or merges in
4. Remove scattered path parsing utilities
