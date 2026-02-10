# Abstract Transformer Topology — Implementation Log

## Decisions

- Embedding LayerInfo uses role="up", unembed uses role="down" — these are arbitrary since embedding/unembed aren't block components. They exist so LayerInfo always has a role. Could use a separate type but it's not worth the complexity.

## Niggles

- `ordered_layers()` preserves `target_module_paths` ordering from ComponentModel. Attn-before-mlp ordering within a block is an unsolved issue, fine for now.
- Partial FFN decomposition (decomposing only attention, not FFN) would crash `_build_ffn`. Not an issue in practice — all current configs decompose both.

## Assumptions

- Every target module path contains exactly one integer segment (the block index) — now asserted
- Role mapping patterns are exhaustive — every target path matches exactly one pattern
- Within a block, attention roles and FFN roles don't overlap

## Terminology

"Role" = the abstract function a layer serves within a transformer block. Maps concrete module names (q_proj, c_attn, c_fc, gate_proj, etc.) onto canonical abstract names (q, k, v, qkv, o, up, down, gate). This lets consumers reason about model structure without knowing architecture-specific naming.
