# App TODO

## LocalAttributionsGraph

- [ ] Test edge highlighting via SVG string vs DOM manipulation
  - Current approach uses `{@html}` for bulk edge rendering (performance), but requires DOM manipulation for highlighting
  - Alternative: Re-render edges reactively as Svelte components
  - Need to benchmark both approaches with large graphs (10k+ edges)

## Node Keys

Node keys follow the format `"layer:seq:cIdx"` where:
- `layer`: The model layer name (e.g., `h.0.attn.q_proj`, `h.2.mlp.c_fc`)
- `seq`: Sequence position (0-indexed)
- `cIdx`: Component index within the layer

### Non-Interventable Nodes

`wte` and `output` are **pseudo-layers** used for visualization only:
- `wte` (word token embedding): Represents the input embeddings - not part of the decomposed model
- `output`: Represents output logits/predictions - not part of the decomposed model

These nodes appear in the attribution graph for visualization but **cannot be intervened on**.
Only internal layers (attn/mlp projections) can have their components selectively activated.

The helper `isInterventableNode()` in `localAttributionsTypes.ts` filters these out.
