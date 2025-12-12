# Plan: Replace nodeImportance with CI Values

## Overview

Replace the derived `nodeImportance` metric (sum of squared edge values) with actual CI values from the model. This enables:
- Ordering nodes by their causal importance (highest CI on left)
- Showing CI values on hover
- Displaying all nodes above the CI threshold, regardless of edge connectivity

## Decisions

- **Node opacity**: Brighter = higher CI (normalized by max CI in graph)
- **CI display format**: "CI: 0.842"
- **Pseudo-layers**: Keep as-is (wte by position, output by probability)
- **Tooltip**: Add position-specific CI to tooltip header. `mean_ci` in activation contexts remains unchanged.
- **Edge filtering**: Nodes appear if CI > threshold, even if isolated (no edges)

## Current State

### Backend
- `node_ci_vals: dict[str, float]` is computed in `compute.py` via `extract_node_ci_vals()`
- Stored in database (`graphs` table, `node_ci_vals` column)
- Filtered by `ci_threshold` in `graphs.py`
- Then converted to `nodeImportance` via `compute_edge_stats()` (sum of squared edge strengths)
- `nodeImportance` is what gets sent to frontend

### Frontend
- `GraphData.nodeImportance` used for:
  - Node opacity calculation
  - Node ordering (when layout="importance")
  - Enumerating which nodes exist
- Hover tooltip shows `mean_ci` from activation contexts summary (a different, component-level metric)

## Implementation Steps

### 1. Backend: Simplify `compute_edge_stats()`

**File: `spd/app/backend/routers/graphs.py`**

- Remove `nodeImportance` computation from `compute_edge_stats()`
- Keep only `maxAbsAttr` calculation (still needed for edge visualization)
- Rename function to `compute_max_abs_attr()` or similar

### 2. Backend: Update `process_edges_for_response()`

**File: `spd/app/backend/routers/graphs.py`**

- Change signature to accept `node_ci_vals` and pass it through
- Return `(edges, node_ci_vals, max_abs_attr)` instead of `(edges, node_importance, max_abs_attr)`

### 3. Backend: Update Response Schema

**File: `spd/app/backend/routers/graphs.py`**

- Change `GraphData` response to use `nodeCiVals` instead of `nodeImportance`
- Update all endpoints that return `GraphData`:
  - POST `/api/graphs`
  - POST `/api/graphs/optimized/stream`
  - GET `/api/graphs` (retrieval)

### 4. Frontend: Update Types

**File: `spd/app/frontend/src/lib/localAttributionsTypes.ts`**

```typescript
export type GraphData = {
    id: number;
    tokens: string[];
    edges: Edge[];
    outputProbs: Record<string, OutputProbEntry>;
    nodeCiVals: Record<string, number>;  // CHANGED from nodeImportance
    maxAbsAttr: number;
    l0_total: number;
    optimization?: OptimizationResult;
};
```

### 5. Frontend: Update LocalAttributionsGraph.svelte

**File: `spd/app/frontend/src/components/LocalAttributionsGraph.svelte`**

- Rename `maxImportance` → `maxCi` (derived from max of `nodeCiVals`)
- Update `allNodes` to derive from `Object.keys(data.nodeCiVals)`
- Update node ordering: sort by `nodeCiVals[key]` descending (highest CI on left)
- Update node opacity: `intensity = ci / maxCi`

### 6. Frontend: Add CI to Tooltip

**File: `spd/app/frontend/src/components/local-attr/NodeTooltip.svelte`**

- Accept `nodeCiVals` as prop
- For component nodes, display "CI: {value}" in the tooltip header
- wte and output nodes don't show CI (they don't have real CI values)

### 7. Frontend: Update InterventionsView.svelte

**File: `spd/app/frontend/src/components/local-attr/InterventionsView.svelte`**

- Replace `nodeImportance` references with `nodeCiVals`

## Files to Modify

| File | Changes |
|------|---------|
| `spd/app/backend/routers/graphs.py` | Simplify `compute_edge_stats()`, update response schema, pass through `node_ci_vals` |
| `spd/app/frontend/src/lib/localAttributionsTypes.ts` | Rename `nodeImportance` → `nodeCiVals` |
| `spd/app/frontend/src/components/LocalAttributionsGraph.svelte` | Update all `nodeImportance` refs, node ordering, opacity |
| `spd/app/frontend/src/components/local-attr/NodeTooltip.svelte` | Add CI display |
| `spd/app/frontend/src/components/local-attr/InterventionsView.svelte` | Update `nodeImportance` refs |
