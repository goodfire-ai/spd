<script lang="ts">
    import * as api from "../lib/api";
    import type { LocalAttributions, PairAttribution } from "../lib/api";

    let data = $state<LocalAttributions | null>(null);
    let loading = $state(false);
    let error = $state<string | null>(null);

    async function loadData() {
        loading = true;
        error = null;
        try {
            data = await api.getLocalAttributions();
        } catch (e) {
            error = e instanceof Error ? e.message : "Unknown error";
        } finally {
            loading = false;
        }
    }

    $effect(() => {
        loadData();
    });

    // Parse layer names to understand structure
    // e.g. "h.0.attn.q_proj" -> { block: 0, type: "attn", subtype: "q_proj" }
    type LayerInfo = {
        name: string;
        block: number;
        type: "attn" | "mlp";
        subtype: string;
    };

    function parseLayerName(name: string): LayerInfo | null {
        const match = name.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
        if (!match) return null;
        return {
            name,
            block: parseInt(match[1]),
            type: match[2] as "attn" | "mlp",
            subtype: match[3],
        };
    }

    // Get unique layers from pairs
    let layers = $derived.by(() => {
        if (!data) return [];
        const layerSet = new Set<string>();
        for (const pair of data.pairs) {
            layerSet.add(pair.source);
            layerSet.add(pair.target);
        }
        return Array.from(layerSet)
            .map(parseLayerName)
            .filter((l): l is LayerInfo => l !== null);
    });

    // Layout constants
    const TOKEN_HEIGHT = 30;
    const LAYER_HEIGHT = 60;
    const SEQ_WIDTH = 80;
    const COMPONENT_RADIUS = 6;
    const MARGIN = { top: 40, right: 20, bottom: 60, left: 100 };

    // Define vertical positions for layer subtypes
    // Attention: q, k, v are parallel, then o above
    // MLP: c_fc, then down_proj above
    const SUBTYPE_Y_OFFSETS: Record<string, number> = {
        q_proj: 0,
        k_proj: 0,
        v_proj: 0,
        o_proj: 1,
        c_fc: 2,
        down_proj: 3,
    };

    // Horizontal offsets within a sequence position for parallel layers
    const SUBTYPE_X_OFFSETS: Record<string, number> = {
        q_proj: -20,
        k_proj: 0,
        v_proj: 20,
        o_proj: 0,
        c_fc: 0,
        down_proj: 0,
    };

    // Colors for different layer types
    const LAYER_COLORS: Record<string, string> = {
        q_proj: "#e91e63",
        k_proj: "#9c27b0",
        v_proj: "#673ab7",
        o_proj: "#3f51b5",
        c_fc: "#009688",
        down_proj: "#4caf50",
    };

    // Get position for a node
    function getNodePos(
        layerName: string,
        seqPos: number,
        componentIdx: number,
    ): { x: number; y: number } | null {
        const info = parseLayerName(layerName);
        if (!info) return null;

        const yOffset = SUBTYPE_Y_OFFSETS[info.subtype] ?? 0;
        const xOffset = SUBTYPE_X_OFFSETS[info.subtype] ?? 0;

        return {
            x: MARGIN.left + seqPos * SEQ_WIDTH + xOffset,
            y: MARGIN.top + (3 - yOffset) * LAYER_HEIGHT, // Invert so higher layers are higher visually
        };
    }

    // Edge type
    type Edge = {
        sourceLayer: string;
        targetLayer: string;
        sourceSeq: number;
        targetSeq: number;
        sourceCIdx: number;
        targetCIdx: number;
        attribution: number;
    };

    // Find max attribution across all data (quick scan)
    let maxAttr = $derived.by(() => {
        if (!data) return 1;
        let max = 0;
        for (const pair of data.pairs) {
            for (const s1 of pair.attribution) {
                for (const c1 of s1) {
                    for (const s2 of c1) {
                        for (const val of s2) {
                            const abs = Math.abs(val);
                            if (abs > max) max = abs;
                        }
                    }
                }
            }
        }
        return max || 1;
    });

    // SVG dimensions
    let svgWidth = $derived(data ? MARGIN.left + data.tokens.length * SEQ_WIDTH + MARGIN.right : 800);
    let svgHeight = $derived(MARGIN.top + 4 * LAYER_HEIGHT + MARGIN.bottom);

    // Filter state
    let minAttrThreshold = $state(0.3);
    let maxEdges = $state(200);

    // Only compute edges that pass threshold, stop early at maxEdges
    let filteredEdges = $derived.by(() => {
        if (!data) return [];
        const result: Edge[] = [];
        const absThreshold = minAttrThreshold * maxAttr;

        outer: for (const pair of data.pairs) {
            const attr = pair.attribution;
            for (let sIn = 0; sIn < attr.length; sIn++) {
                for (let cInIdx = 0; cInIdx < attr[sIn].length; cInIdx++) {
                    for (let sOut = 0; sOut < attr[sIn][cInIdx].length; sOut++) {
                        for (let cOutIdx = 0; cOutIdx < attr[sIn][cInIdx][sOut].length; cOutIdx++) {
                            const val = attr[sIn][cInIdx][sOut][cOutIdx];
                            if (Math.abs(val) >= absThreshold) {
                                result.push({
                                    sourceLayer: pair.source,
                                    targetLayer: pair.target,
                                    sourceSeq: sIn,
                                    targetSeq: sOut,
                                    sourceCIdx: pair.trimmed_c_in_idxs[cInIdx],
                                    targetCIdx: pair.trimmed_c_out_idxs[cOutIdx],
                                    attribution: val,
                                });
                                if (result.length >= maxEdges * 2) break outer; // Get 2x, then sort & trim
                            }
                        }
                    }
                }
            }
        }
        // Sort by magnitude, take top N
        result.sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution));
        return result.slice(0, maxEdges);
    });

    let totalEdgeEstimate = $derived.by(() => {
        if (!data) return 0;
        let count = 0;
        for (const pair of data.pairs) {
            count +=
                pair.attribution.length *
                (pair.attribution[0]?.length ?? 0) *
                (pair.attribution[0]?.[0]?.length ?? 0) *
                (pair.attribution[0]?.[0]?.[0]?.length ?? 0);
        }
        return count;
    });
</script>

<div class="local-attributions">
    <h2>Local Attributions Graph</h2>

    {#if loading}
        <div class="loading">Loading...</div>
    {:else if error}
        <div class="error">{error}</div>
    {:else if data}
        <div class="controls">
            <label>
                Min threshold:
                <input type="range" min="0" max="1" step="0.01" bind:value={minAttrThreshold} />
                {(minAttrThreshold * 100).toFixed(0)}%
            </label>
            <label>
                Max edges:
                <input type="number" min="10" max="5000" step="100" bind:value={maxEdges} />
            </label>
            <span class="edge-count">Showing {filteredEdges.length} edges (~{totalEdgeEstimate.toLocaleString()} total)</span>
        </div>

        <div class="graph-container">
            <svg width={svgWidth} height={svgHeight}>
                <!-- Layer labels on the left -->
                <g class="layer-labels">
                    {#each Object.entries(SUBTYPE_Y_OFFSETS) as [subtype, yOffset]}
                        <text
                            x={MARGIN.left - 10}
                            y={MARGIN.top + (3 - yOffset) * LAYER_HEIGHT + 4}
                            text-anchor="end"
                            class="layer-label"
                            fill={LAYER_COLORS[subtype]}
                        >
                            {subtype}
                        </text>
                    {/each}
                </g>

                <!-- Edges -->
                <g class="edges">
                    {#each filteredEdges as edge}
                        {@const sourcePos = getNodePos(edge.sourceLayer, edge.sourceSeq, edge.sourceCIdx)}
                        {@const targetPos = getNodePos(edge.targetLayer, edge.targetSeq, edge.targetCIdx)}
                        {#if sourcePos && targetPos}
                            <line
                                x1={sourcePos.x}
                                y1={sourcePos.y}
                                x2={targetPos.x}
                                y2={targetPos.y}
                                stroke={edge.attribution > 0 ? "#2196f3" : "#f44336"}
                                stroke-width={Math.max(0.5, (Math.abs(edge.attribution) / maxAttr) * 3)}
                                opacity={0.3 + (Math.abs(edge.attribution) / maxAttr) * 0.7}
                            />
                        {/if}
                    {/each}
                </g>

                <!-- Nodes for each layer at each sequence position -->
                <g class="nodes">
                    {#each layers as layer}
                        {#each data.tokens as _, seqIdx}
                            {@const pos = getNodePos(layer.name, seqIdx, 0)}
                            {#if pos}
                                <circle
                                    cx={pos.x}
                                    cy={pos.y}
                                    r={COMPONENT_RADIUS}
                                    fill={LAYER_COLORS[layer.subtype] ?? "#999"}
                                    stroke="white"
                                    stroke-width="1"
                                />
                            {/if}
                        {/each}
                    {/each}
                </g>

                <!-- Token labels at the bottom -->
                <g class="token-labels">
                    {#each data.tokens as token, i}
                        <text
                            x={MARGIN.left + i * SEQ_WIDTH}
                            y={svgHeight - MARGIN.bottom + 30}
                            text-anchor="middle"
                            class="token-label"
                        >
                            {token}
                        </text>
                        <text
                            x={MARGIN.left + i * SEQ_WIDTH}
                            y={svgHeight - MARGIN.bottom + 45}
                            text-anchor="middle"
                            class="seq-idx"
                        >
                            [{i}]
                        </text>
                    {/each}
                </g>

                <!-- Vertical guides for sequence positions -->
                <g class="guides">
                    {#each data.tokens as _, i}
                        <line
                            x1={MARGIN.left + i * SEQ_WIDTH}
                            y1={MARGIN.top - 10}
                            x2={MARGIN.left + i * SEQ_WIDTH}
                            y2={svgHeight - MARGIN.bottom + 10}
                            stroke="#eee"
                            stroke-dasharray="2,2"
                        />
                    {/each}
                </g>
            </svg>
        </div>

        <div class="legend">
            <h3>Legend</h3>
            <div class="legend-items">
                {#each Object.entries(LAYER_COLORS) as [subtype, color]}
                    <div class="legend-item">
                        <span class="legend-dot" style="background: {color}"></span>
                        {subtype}
                    </div>
                {/each}
            </div>
            <div class="legend-edge">
                <span class="edge-pos"></span> Positive attribution
                <span class="edge-neg"></span> Negative attribution
            </div>
        </div>
    {/if}
</div>

<style>
    .local-attributions {
        padding: 1rem;
    }

    h2 {
        margin-top: 0;
    }

    .loading,
    .error {
        padding: 1rem;
        border-radius: 4px;
    }

    .loading {
        background: #e3f2fd;
        color: #1565c0;
    }

    .error {
        background: #ffebee;
        color: #c62828;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem 1rem;
        background: #f5f5f5;
        border-radius: 4px;
    }

    .controls label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .controls input[type="range"] {
        width: 150px;
    }

    .controls input[type="number"] {
        width: 80px;
        padding: 0.25rem;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .edge-count {
        color: #666;
        font-size: 0.9rem;
    }

    .graph-container {
        overflow: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
    }

    svg {
        display: block;
    }

    .layer-label {
        font-size: 12px;
        font-weight: 500;
    }

    .token-label {
        font-size: 12px;
        font-family: monospace;
        font-weight: 500;
    }

    .seq-idx {
        font-size: 10px;
        fill: #999;
    }

    .legend {
        margin-top: 1rem;
        padding: 1rem;
        background: #f9f9f9;
        border-radius: 4px;
    }

    .legend h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }

    .legend-items {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.85rem;
    }

    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }

    .legend-edge {
        margin-top: 0.5rem;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .edge-pos,
    .edge-neg {
        display: inline-block;
        width: 30px;
        height: 3px;
        margin-right: 0.25rem;
    }

    .edge-pos {
        background: #2196f3;
    }

    .edge-neg {
        background: #f44336;
    }
</style>
