# Implementation Plan: Output Attribution View

## Overview

Add an "attribution mode" dropdown to the graph visualization that allows switching between:
- **"attr to connected"** (default) - current behavior showing edges between adjacent layers
- **"attr to output"** - new edges showing total attribution from each node to final-position output logits

When "attr to output" is selected, the graph displays different edges: direct connections from each internal node to output nodes, where edge value = summed attribution through all paths.

## Background Context

### Current System

The app displays attribution graphs for neural network component decompositions:

- **Nodes**: Components at each layer/sequence position, identified by key `"layer:seq:cIdx"`
- **Edges**: Attribution values between adjacent layers (`edge.val = gradient × source_activation`)
- **Output nodes**: Special pseudo-layer "output" showing predicted tokens (prob > threshold)
- **Data structures** (in `localAttributionsTypes.ts`):
  ```typescript
  type Edge = { src: string; tgt: string; val: number };
  type GraphData = {
      edges: Edge[];
      edgesBySource: Map<string, Edge[]>;
      edgesByTarget: Map<string, Edge[]>;
      outputProbs: Record<string, OutputProbEntry>;  // "seq:cIdx" -> {prob, token}
      // ... other fields
  };
  ```

### Output Attribution Concept

For "attr to output" mode, we compute total attribution from each node to each output logit at the **final sequence position only**. This sums all path contributions:

```
output_attr[node → output] = Σ (product of edge values along path)
                              for all paths from node to output
```

This can be computed efficiently via dynamic programming (backward pass from outputs).

## Implementation Steps

### Step 1: Backend - Add `compute_output_attributions()` function

**File**: `spd/app/backend/compute.py`

Add a new function that computes output attribution edges from existing connected edges:

```python
def compute_output_attributions(
    edges: list[Edge],
    output_probs: dict[str, OutputProbEntry],
    final_seq_pos: int,
) -> list[Edge]:
    """
    Compute total attribution from each node to output logits at final position.

    Algorithm:
    1. Build adjacency map: node -> [(child_node, edge_val), ...]
    2. Identify output nodes at final_seq_pos
    3. For each output node, run backward DP to compute total attribution from all nodes
    4. Return new edges: internal_node -> output_node with summed attribution

    Args:
        edges: Connected attribution edges (adjacent layers)
        output_probs: Output probability entries keyed by "seq:cIdx"
        final_seq_pos: The final sequence position (len(tokens) - 1)

    Returns:
        List of edges from each internal node to each output node at final position
    """
```

**Algorithm detail**:
```python
# Build adjacency: source -> list of (target, strength)
adjacency = defaultdict(list)
all_nodes = set()
for edge in edges:
    adjacency[edge.source_key].append((edge.target_key, edge.strength))
    all_nodes.add(edge.source_key)
    all_nodes.add(edge.target_key)

# Find output nodes at final position
output_nodes = [
    f"output:{final_seq_pos}:{cIdx}"
    for key, entry in output_probs.items()
    if key.startswith(f"{final_seq_pos}:")
]

# For each output, compute attribution from all nodes via memoized DFS
new_edges = []
for output_node in output_nodes:
    memo = {output_node: 1.0}

    def get_output_attr(node: str) -> float:
        if node in memo:
            return memo[node]
        total = 0.0
        for child, edge_val in adjacency.get(node, []):
            total += edge_val * get_output_attr(child)
        memo[node] = total
        return total

    # Compute attribution for all non-output nodes
    for node in all_nodes:
        if node.startswith("output:"):
            continue
        attr = get_output_attr(node)
        if attr != 0:
            new_edges.append(Edge(
                source=parse_node_key(node),
                target=parse_node_key(output_node),
                strength=attr,
                is_cross_seq=False
            ))

return new_edges
```

**Helper needed**: `parse_node_key(key: str) -> Node` to convert "layer:seq:cIdx" back to Node dataclass.

### Step 2: Backend - Extend graph retrieval endpoint

**File**: `spd/app/backend/routers/graphs.py`

Modify `get_graphs()` endpoint to accept an `attribution_mode` parameter:

```python
@router.get("/{prompt_id}")
async def get_graphs(
    prompt_id: int,
    normalize: NormalizeType = "none",
    ci_threshold: float = 0.0,
    attribution_mode: Literal["connected", "output"] = "connected",  # NEW
    state: AppState = Depends(get_app_state),
) -> list[GraphResponse]:
```

When `attribution_mode == "output"`:
1. Retrieve the stored graph with connected edges as usual
2. Call `compute_output_attributions()` to get output attribution edges
3. Replace edges in the response with the new output attribution edges
4. Recompute `maxAbsAttr` for the new edges

**Note**: Output attribution edges are computed on-the-fly, not stored. This keeps storage simple and ensures they're always derived from the source connected edges.

### Step 3: Backend - Update schema

**File**: `spd/app/backend/schemas.py`

Add type alias if not using inline Literal:
```python
AttributionMode = Literal["connected", "output"]
```

### Step 4: Frontend - Update API client

**File**: `spd/app/frontend/src/lib/api/graphs.ts`

Add `attributionMode` parameter to `getGraphs()`:

```typescript
export type AttributionMode = "connected" | "output";

export async function getGraphs(
    promptId: number,
    normalize: NormalizeType,
    ciThreshold: number,
    attributionMode: AttributionMode = "connected",  // NEW
): Promise<GraphData[]> {
    const params = new URLSearchParams({
        normalize,
        ci_threshold: ciThreshold.toString(),
        attribution_mode: attributionMode,  // NEW
    });
    // ... rest unchanged
}
```

Also update `lib/api/index.ts` to export the new type.

### Step 5: Frontend - Add to ViewSettings type

**File**: `spd/app/frontend/src/components/local-attr/types.ts`

Add `attributionMode` to `ViewSettings`:

```typescript
import type { AttributionMode } from "../../lib/api";

export type ViewSettings = {
    topK: number;
    componentGap: number;
    layerGap: number;
    normalizeEdges: NormalizeType;
    ciThreshold: number;
    attributionMode: AttributionMode;  // NEW
};
```

Update `defaultViewSettings` in `LocalAttributionsTab.svelte`:
```typescript
const defaultViewSettings: ViewSettings = {
    // ... existing
    attributionMode: "connected",  // NEW
};
```

### Step 6: Frontend - Add dropdown to ViewControls

**File**: `spd/app/frontend/src/components/local-attr/ViewControls.svelte`

Add a new dropdown next to the "Edge Norm" dropdown:

```svelte
<script lang="ts">
    import type { AttributionMode } from "../../lib/api";

    type Props = {
        // ... existing props
        attributionMode: AttributionMode;  // NEW
        onAttributionModeChange: (mode: AttributionMode) => void;  // NEW
    };
</script>

<!-- In the template, add before or after Edge Norm dropdown -->
<label>
    <span>Attribution</span>
    <select
        value={attributionMode}
        onchange={(e) => onAttributionModeChange(e.currentTarget.value as AttributionMode)}
    >
        <option value="connected">attr to connected</option>
        <option value="output">attr to output</option>
    </select>
</label>
```

### Step 7: Frontend - Handle mode changes in LocalAttributionsTab

**File**: `spd/app/frontend/src/components/LocalAttributionsTab.svelte`

Add handler for attribution mode changes (similar to `handleNormalizeChange`):

```typescript
async function handleAttributionModeChange(mode: AttributionMode) {
    if (!activeGraph || !activeCard) return;

    // Update view settings
    const updatedGraph = {
        ...activeGraph,
        viewSettings: { ...activeGraph.viewSettings, attributionMode: mode },
    };

    // Refetch graph data with new mode
    refetchingGraphId = activeGraph.id;
    try {
        const [refetched] = await api.getGraphs(
            activeCard.id,
            updatedGraph.viewSettings.normalizeEdges,
            updatedGraph.viewSettings.ciThreshold,
            mode,  // NEW parameter
        );
        // ... update graph data similar to handleNormalizeChange
    } finally {
        refetchingGraphId = null;
    }
}
```

Pass the new props to `ViewControls`:
```svelte
<ViewControls
    // ... existing props
    attributionMode={activeGraph.viewSettings.attributionMode}
    onAttributionModeChange={handleAttributionModeChange}
/>
```

Also update the `getGraphs()` calls in other places to pass the attribution mode:
- `addPromptCard()`
- `handleCiThresholdChange()`
- `handleNormalizeChange()`

### Step 8: Update InterventionsView if needed

**File**: `spd/app/frontend/src/components/local-attr/InterventionsView.svelte`

If `InterventionsView` also has `ViewControls`, add the same attribution mode props there.

## File Summary

| File | Changes |
|------|---------|
| `backend/compute.py` | Add `compute_output_attributions()` function |
| `backend/routers/graphs.py` | Add `attribution_mode` parameter to `get_graphs()` |
| `backend/schemas.py` | Add `AttributionMode` type (optional) |
| `frontend/src/lib/api/graphs.ts` | Add `attributionMode` parameter, export type |
| `frontend/src/lib/api/index.ts` | Re-export `AttributionMode` type |
| `frontend/src/components/local-attr/types.ts` | Add `attributionMode` to `ViewSettings` |
| `frontend/src/components/local-attr/ViewControls.svelte` | Add attribution mode dropdown |
| `frontend/src/components/LocalAttributionsTab.svelte` | Add handler, update API calls |
| `frontend/src/components/local-attr/InterventionsView.svelte` | Pass through attribution mode (if applicable) |

## Testing

1. Load a run and compute a graph
2. Verify default "attr to connected" shows current edges
3. Switch to "attr to output" and verify:
   - Edges now go from internal nodes directly to output nodes at final position
   - Edge colors reflect attribution magnitude
   - Pinning output nodes + "hide unpinned edges" filters appropriately
4. Switch back to "attr to connected" and verify original edges return
5. Test with agents via API: `GET /api/graphs/{prompt_id}?attribution_mode=output`

## Notes

- Output attribution edges are computed on-the-fly, not stored in the database
- The `wte` (embedding) nodes will have small attribution values due to many hops - this is expected
- Cross-sequence attention edges are handled naturally by the path-sum algorithm
- Only final sequence position outputs are included (not all positions)
