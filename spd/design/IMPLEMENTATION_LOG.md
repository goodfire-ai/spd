# Abstract Transformer Topology — Implementation Log

## Decisions

- Embedding LayerInfo uses role="up", unembed uses role="down" — these are arbitrary since embedding/unembed aren't block components. They exist so LayerInfo always has a role. Could use a separate type but it's not worth the complexity.

## Niggles

- `ordered_paths()` returns paths in block order (attn before ffn within each block). This matches the execution order for most models but isn't verified against the actual `target_module_paths` ordering from `ComponentModel`. If a model has a different ordering convention, this could diverge.
- Not all target modules may be decomposed (partial decomposition). The topology only includes paths that are in `target_module_paths`, so a block might have attention but no FFN if the FFN wasn't decomposed. Currently `_build_ffn` asserts up+down exist, which would fail for partial FFN decomposition.

## Assumptions

- Every target module path contains exactly one integer segment (the block index)
- Role mapping patterns are exhaustive — every target path matches exactly one pattern
- Within a block, attention roles and FFN roles don't overlap (same role name can't be both)
