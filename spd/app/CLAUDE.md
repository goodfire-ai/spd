# SPD App

Web-based visualization and analysis tool for exploring neural network component decompositions.

- **Backend**: Python FastAPI (`backend/`)
- **Frontend**: Svelte 5 + TypeScript (`frontend/`)
- **Database**: SQLite at `~/.spd/local_attr.db`

## Project Context

This is a **rapidly iterated research tool**. Key implications:

- **No backwards compatibility**: Schema changes don't need migrations
- **Database is disposable**: Delete `~/.spd/local_attr.db` if schema changes break things
- **Prefer simplicity**: Avoid over-engineering for hypothetical future needs

## Running the App

```bash
python -m spd.app.run_app
```

This launches both backend (FastAPI/uvicorn) and frontend (Vite) dev servers.

---

## Architecture Overview

### Backend Structure

```
backend/
├── server.py              # FastAPI app, CORS, routers
├── state.py               # Singleton StateManager (loaded model, tokenizer, etc.)
├── compute.py             # Core attribution computation
├── schemas.py             # Pydantic API models
├── dependencies.py        # FastAPI dependency injection
├── db/database.py         # SQLite interface
├── lib/
│   ├── activation_contexts.py    # Component firing pattern analysis
│   └── component_correlations.py # Co-occurrence metrics
├── optim_cis/
│   └── run_optim_cis.py   # Sparse CI optimization
└── routers/
    ├── runs.py            # Load W&B runs
    ├── graphs.py          # Compute attribution graphs
    ├── prompts.py         # Prompt management
    ├── activation_contexts.py
    └── intervention.py    # Selective component activation
```

### Frontend Structure

```
frontend/src/
├── App.svelte
├── lib/
│   ├── api.ts                    # Main API client
│   ├── localAttributionsApi.ts   # Attribution-specific API
│   ├── localAttributionsTypes.ts # TypeScript types
│   └── interventionTypes.ts
└── components/
    ├── LocalAttributionsTab.svelte    # Main container
    ├── LocalAttributionsGraph.svelte  # SVG graph visualization
    └── local-attr/
        ├── InterventionsView.svelte   # Selective activation UI
        ├── StagedNodesPanel.svelte    # Pinned nodes list
        ├── NodeTooltip.svelte         # Hover card
        ├── ComponentNodeCard.svelte   # Component details
        ├── CorrelationTable.svelte    # Co-occurrence display
        ├── PromptPicker.svelte
        ├── ViewControls.svelte
        ├── graphUtils.ts              # Layout helpers
        └── types.ts                   # UI state types
```

---

## Key Data Structures

### Node Keys

Node keys follow the format `"layer:seq:cIdx"` where:
- `layer`: Model layer name (e.g., `h.0.attn.q_proj`, `h.2.mlp.c_fc`)
- `seq`: Sequence position (0-indexed)
- `cIdx`: Component index within the layer

### Non-Interventable Nodes

`wte` and `output` are **pseudo-layers** for visualization only:
- `wte` (word token embedding): Input embeddings, single pseudo-component (idx 0)
- `output`: Output logits, component_idx = token_id

These appear in attribution graphs but **cannot be intervened on**.
Only internal layers (attn/mlp projections) support selective activation.

Helper: `isInterventableNode()` in `localAttributionsTypes.ts`

### Backend Types (`compute.py`)

```python
Node(layer: str, seq_pos: int, component_idx: int)

Edge(source: Node, target: Node, strength: float, is_cross_seq: bool)
# strength = gradient * activation
# is_cross_seq = True for k/v → o_proj (attention pattern)

LocalAttributionResult(edges: list[Edge], output_probs: Tensor[seq, vocab])
```

### Frontend Types (`localAttributionsTypes.ts`)

```typescript
GraphData = {
  id: number,
  tokens: string[],
  edges: Edge[],                              // {src, tgt, val}
  outputProbs: Record<string, OutputProbEntry>, // "seq:cIdx" → {prob, token}
  nodeImportance: Record<string, number>,     // node_key → Σ(edge²)
  maxAbsAttr: number,
  optimization?: OptimizationResult
}

StoredGraph = {
  id: string,           // Client-generated
  dbId: number,         // Server-side ID
  data: GraphData,
  composerSelection: Set<string>,  // Selected nodes for intervention
  interventionRuns: InterventionRunSummary[]
}

PromptCard = {
  id: string,
  promptId: number,     // DB reference
  tokens: string[],
  graphs: StoredGraph[],
  activeGraphId: string,
  activeView: "graph" | "interventions"
}
```

---

## Core Computations

### Attribution Graph (`compute.py`)

**Entry points**:
- `compute_local_attributions()` - Uses model's natural CI values
- `compute_local_attributions_optimized()` - Sparse CI optimization

**Algorithm** (`compute_edges_from_ci`):
1. Forward pass with CI masks → component activations cached
2. For each target layer, for each alive (seq_pos, component):
   - Compute gradient of target w.r.t. all source layers
   - `strength = grad * source_activation`
   - Create Edge for each alive source component

**Cross-sequence edges**: `is_kv_to_o_pair()` detects k/v → o_proj in same attention block.
These have gradients across sequence positions (causal attention pattern).

### Causal Importance (CI)

CI determines which components are "alive":
- Computed via `model.calc_causal_importances()`
- Thresholded: `ci >= ci_threshold` → active
- For output layer: `prob >= output_prob_threshold`

### CI Optimization (`optim_cis/run_optim_cis.py`)

Finds sparse CI mask that:
- Preserves prediction of target `label_token`
- Minimizes L0 (active component count)
- Uses importance minimality + CE loss

### Intervention Forward

`compute_intervention_forward()`:
1. Build component masks (all zeros)
2. Set mask=1.0 for selected nodes
3. Forward pass → top-k predictions per position

---

## Data Flow

### Run Loading
```
POST /api/runs/load(wandb_path)
  → Load ComponentModel + tokenizer from W&B
  → Build sources_by_target (valid gradient paths)
  → Store in StateManager singleton
  ← LoadedRun
```

### Graph Computation (SSE streaming)
```
POST /api/graphs
  → compute_local_attributions()
  → Stream progress: {type: "progress", current, total, stage}
  ← {type: "complete", data: GraphData}
```

### Intervention
```
POST /api/intervention {text, nodes: ["h.0.attn.q_proj:3:5", ...]}
  → compute_intervention_forward()
  ← InterventionResponse with top-k predictions
```

---

## Database Schema

Located at `~/.spd/local_attr.db`. Delete this file if schema changes cause issues.

| Table | Key | Purpose |
|-------|-----|---------|
| `runs` | `wandb_path` | W&B run references |
| `activation_contexts` | `(run_id, context_length)` | Component firing patterns, gzip-compressed |
| `prompts` | `(run_id, context_length)` | Token sequences |
| `component_activations` | `(prompt_id, component_key)` | Inverted index: component → prompts where it fires |
| `cached_graphs` | `(prompt_id, optimization_params)` | Attribution edges + output probs (gzipped) |
| `intervention_runs` | `cached_graph_id` | Saved intervention results |

---

## State Management

### Backend (`state.py`)

```python
StateManager.get() → AppState:
  - db: LocalAttrDB (always available)
  - run_state: RunState | None
      - model: ComponentModel
      - tokenizer: PreTrainedTokenizerBase
      - sources_by_target: dict[target_layer → source_layers]
      - config, train_loader, context_length
      - activation_contexts_cache: ModelActivationContexts | None
```

### Frontend (`LocalAttributionsTab.svelte`)

- `promptCards` - All open prompt analysis cards
- `activeCard` / `activeGraph` - Current selection
- `pinnedNodes` - Highlighted nodes for tracing
- `componentDetailsCache` - Lazy-loaded component info

---

## Performance Notes

- **Edge limit**: `GLOBAL_EDGE_LIMIT = 5000` in graph visualization
- **SSE streaming**: Long computations stream progress updates
- **Lazy loading**: Component details fetched on hover/pin
- **Compression**: gzip for edges/probs in SQLite
- **Caching**: Graphs cached by (prompt_id, optimization_params)

---

## TODO

### LocalAttributionsGraph

- [ ] Test edge highlighting via SVG string vs DOM manipulation
  - Current approach uses `{@html}` for bulk edge rendering (performance), but requires DOM manipulation for highlighting
  - Alternative: Re-render edges reactively as Svelte components
  - Need to benchmark both approaches with large graphs (10k+ edges)
