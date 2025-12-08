<script lang="ts">
    import type { Intervention } from "./types";
    import type { ActivationContextsSummary, ComponentDetail, PinnedNode } from "../../lib/localAttributionsTypes";
    import { colors } from "../../lib/colors";
    import ComponentDetailCard from "./ComponentDetailCard.svelte";

    // Constants for mini graph
    const COMPONENT_SIZE = 8;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 10, right: 20, bottom: 10, left: 20 };
    const LABEL_WIDTH = 80;
    const COL_PADDING = 12;
    const MIN_COL_WIDTH = 80;
    const COMPONENT_GAP = 6;
    const LAYER_GAP = 24;

    // Logits table constants
    const LOGITS_ROW_HEIGHT = 28;
    const LOGITS_HEADER_HEIGHT = 44;
    const MAX_PREDICTIONS = 5;

    // Row order for layout
    const ROW_ORDER = ["wte", "qkv", "o_proj", "c_fc", "down_proj", "output"];
    const QKV_SUBTYPES = ["q_proj", "k_proj", "v_proj"];

    type Props = {
        interventions: Intervention[];
        tokens: string[];
        activationContextsSummary: ActivationContextsSummary | null;
        componentDetailsCache: Record<string, ComponentDetail>;
        componentDetailsLoading: Record<string, boolean>;
        onLoadComponentDetail: (layer: string, cIdx: number) => void;
        onClear: () => void;
    };

    let {
        interventions,
        tokens,
        activationContextsSummary,
        componentDetailsCache,
        componentDetailsLoading,
        onLoadComponentDetail,
        onClear,
    }: Props = $props();

    function formatProb(prob: number): string {
        if (prob >= 0.01) return (prob * 100).toFixed(1) + "%";
        return (prob * 100).toExponential(1) + "%";
    }

    function formatTime(timestamp: number): string {
        return new Date(timestamp).toLocaleTimeString();
    }

    function getProbBgColor(prob: number): string {
        const { r, g, b } = colors.outputBase;
        const opacity = Math.min(0.7, prob * 0.8 + 0.05);
        return `rgba(${r},${g},${b},${opacity})`;
    }

    function getProbTextColor(prob: number): string {
        return prob > 0.3 ? "white" : colors.textPrimary;
    }

    // Mini graph helpers
    type LayerInfo = { name: string; block: number; type: string; subtype: string };

    function parseLayer(name: string): LayerInfo {
        if (name === "wte") {
            return { name, block: -1, type: "embed", subtype: "wte" };
        }
        if (name === "output") {
            return { name, block: Infinity, type: "output", subtype: "output" };
        }
        const m = name.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
        if (!m) return { name, block: 0, type: "unknown", subtype: name };
        return { name, block: +m[1], type: m[2], subtype: m[3] };
    }

    function getRowKey(layer: string): string {
        const info = parseLayer(layer);
        if (QKV_SUBTYPES.includes(info.subtype)) {
            return `h.${info.block}.qkv`;
        }
        return layer;
    }

    function computeLayout(nodes: PinnedNode[], numTokens: number) {
        // Group nodes by layer and seq position
        const nodesPerLayerSeq: Record<string, number[]> = {};
        const allLayers = new Set<string>();
        const allRows = new Set<string>();

        for (const node of nodes) {
            allLayers.add(node.layer);
            allRows.add(getRowKey(node.layer));
            const key = `${node.layer}:${node.seqIdx}`;
            if (!nodesPerLayerSeq[key]) nodesPerLayerSeq[key] = [];
            nodesPerLayerSeq[key].push(node.cIdx);
        }

        // Sort rows
        const parseRow = (r: string) => {
            if (r === "wte") return { block: -1, subtype: "wte" };
            if (r === "output") return { block: Infinity, subtype: "output" };
            const mQkv = r.match(/h\.(\d+)\.qkv/);
            if (mQkv) return { block: +mQkv[1], subtype: "qkv" };
            const m = r.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
            if (!m) return { block: 0, subtype: r };
            return { block: +m[1], subtype: m[3] };
        };

        const rows = Array.from(allRows).sort((a, b) => {
            const infoA = parseRow(a);
            const infoB = parseRow(b);
            if (infoA.block !== infoB.block) return infoA.block - infoB.block;
            const idxA = ROW_ORDER.indexOf(infoA.subtype);
            const idxB = ROW_ORDER.indexOf(infoB.subtype);
            return idxA - idxB;
        });

        // Calculate column widths based on content
        const maxComponentsPerSeq = Array.from({ length: numTokens }, (_, seqIdx) => {
            let maxAtSeq = 0;
            for (const row of rows) {
                if (row.endsWith(".qkv")) {
                    const blockMatch = row.match(/h\.(\d+)/);
                    if (blockMatch) {
                        const block = blockMatch[1];
                        let totalQkv = 0;
                        for (const subtype of QKV_SUBTYPES) {
                            const layer = `h.${block}.attn.${subtype}`;
                            const layerNodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                            totalQkv += layerNodes.length;
                        }
                        totalQkv += 2;
                        maxAtSeq = Math.max(maxAtSeq, totalQkv);
                    }
                } else {
                    for (const layer of allLayers) {
                        if (getRowKey(layer) === row) {
                            const layerNodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                            maxAtSeq = Math.max(maxAtSeq, layerNodes.length);
                        }
                    }
                }
            }
            return maxAtSeq;
        });

        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + COMPONENT_GAP) + COL_PADDING * 2)
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Calculate logits section height
        const logitsHeight = LOGITS_HEADER_HEIGHT + MAX_PREDICTIONS * LOGITS_ROW_HEIGHT;

        // Assign Y positions for graph (below logits)
        const graphTopY = logitsHeight + 20; // gap between logits and graph
        const rowYPositions: Record<string, number> = {};
        let currentY = graphTopY;
        for (const row of rows.slice().reverse()) {
            rowYPositions[row] = currentY;
            currentY += COMPONENT_SIZE + LAYER_GAP;
        }

        const layerYPositions: Record<string, number> = {};
        for (const layer of allLayers) {
            const rowKey = getRowKey(layer);
            layerYPositions[layer] = rowYPositions[rowKey];
        }

        // Position nodes
        const nodePositions: Record<string, { x: number; y: number }> = {};

        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const isQkv = QKV_SUBTYPES.includes(info.subtype);

            for (let seqIdx = 0; seqIdx < numTokens; seqIdx++) {
                const layerNodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!layerNodes) continue;

                let baseX = seqXStarts[seqIdx] + COL_PADDING;
                const baseY = layerYPositions[layer];

                if (isQkv) {
                    const subtypeIdx = QKV_SUBTYPES.indexOf(info.subtype);
                    for (let i = 0; i < subtypeIdx; i++) {
                        const prevLayer = `h.${info.block}.attn.${QKV_SUBTYPES[i]}`;
                        const prevNodes = nodesPerLayerSeq[`${prevLayer}:${seqIdx}`]?.length ?? 0;
                        baseX += prevNodes * (COMPONENT_SIZE + COMPONENT_GAP);
                        baseX += COMPONENT_SIZE + COMPONENT_GAP;
                    }
                }

                layerNodes.forEach((cIdx, i) => {
                    nodePositions[`${layer}:${seqIdx}:${cIdx}`] = {
                        x: baseX + i * (COMPONENT_SIZE + COMPONENT_GAP) + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                });
            }
        }

        const totalSeqWidth = seqXStarts[seqXStarts.length - 1] + seqWidths[seqWidths.length - 1];
        const width = totalSeqWidth + MARGIN.right;
        const maxY = rows.length > 0 ? Math.max(...Object.values(layerYPositions)) + COMPONENT_SIZE : graphTopY;
        const height = maxY + MARGIN.bottom + 40; // extra space for token labels

        return {
            nodePositions,
            layerYPositions,
            seqWidths,
            seqXStarts,
            width,
            height,
            logitsHeight,
            graphTopY,
            nodesPerLayerSeq,
            allLayers,
        };
    }

    // Hover state
    let hoveredNode = $state<{ layer: string; seqIdx: number; cIdx: number } | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state({ x: 0, y: 0 });
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    function handleNodeMouseEnter(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        hoveredNode = { layer, seqIdx, cIdx };
        tooltipPos = calcTooltipPos(event.clientX, event.clientY);

        if (layer !== "output" && activationContextsSummary) {
            onLoadComponentDetail(layer, cIdx);
        }
    }

    function handleNodeMouseLeave() {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        hoverTimeout = setTimeout(() => {
            if (!isHoveringTooltip) {
                hoveredNode = null;
            }
            hoverTimeout = null;
        }, 50);
    }

    function calcTooltipPos(mouseX: number, mouseY: number) {
        const padding = 15;
        let left = mouseX + padding;
        let top = mouseY + padding;
        if (typeof window !== "undefined") {
            if (left + 400 > window.innerWidth) left = mouseX - 400 - padding;
            if (top + 300 > window.innerHeight) top = mouseY - 300 - padding;
        }
        return { x: Math.max(0, left), y: Math.max(0, top) };
    }

    function getRowLabel(layer: string): string {
        const info = parseLayer(layer);
        const rowKey = getRowKey(layer);
        if (rowKey.endsWith(".qkv")) return `${info.block}.q/k/v`;
        if (layer === "wte" || layer === "output") return layer;
        return `${info.block}.${info.subtype}`;
    }
</script>

<div class="interventions-view">
    {#if interventions.length === 0}
        <div class="empty-state">
            <p>No interventions yet.</p>
            <p class="hint">Stage nodes from the Graph view and click "Run Intervention"</p>
        </div>
    {:else}
        <div class="interventions-header">
            <span>{interventions.length} intervention{interventions.length === 1 ? "" : "s"}</span>
            <button class="clear-btn" onclick={onClear}>Clear All</button>
        </div>

        <div class="interventions-list">
            {#each interventions as intervention, idx (intervention.id)}
                {@const layout = computeLayout(intervention.nodes, intervention.result.input_tokens.length)}
                <div class="intervention-card">
                    <div class="intervention-header">
                        <span class="intervention-title">Intervention #{idx + 1}</span>
                        <span class="intervention-time">{formatTime(intervention.timestamp)}</span>
                    </div>

                    <!-- Unified visualization -->
                    <div class="unified-viz-wrapper">
                        <!-- Labels column -->
                        <div class="labels-column" style="width: {LABEL_WIDTH}px;">
                            <svg width={LABEL_WIDTH} height={layout.height} style="display: block;">
                                <!-- Logits labels -->
                                <text
                                    x={LABEL_WIDTH - 8}
                                    y={LOGITS_HEADER_HEIGHT / 2}
                                    text-anchor="end"
                                    dominant-baseline="middle"
                                    font-size="10"
                                    font-weight="600"
                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                    fill={colors.textSecondary}
                                >
                                    Logits
                                </text>
                                {#each Array(MAX_PREDICTIONS) as _, rank (rank)}
                                    <text
                                        x={LABEL_WIDTH - 8}
                                        y={LOGITS_HEADER_HEIGHT + rank * LOGITS_ROW_HEIGHT + LOGITS_ROW_HEIGHT / 2}
                                        text-anchor="end"
                                        dominant-baseline="middle"
                                        font-size="9"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textMuted}
                                    >
                                        #{rank + 1}
                                    </text>
                                {/each}

                                <!-- Graph layer labels -->
                                {#each Object.entries(layout.layerYPositions) as [layer, y] (layer)}
                                    {@const yCenter = y + COMPONENT_SIZE / 2}
                                    <text
                                        x={LABEL_WIDTH - 8}
                                        y={yCenter}
                                        text-anchor="end"
                                        dominant-baseline="middle"
                                        font-size="10"
                                        font-weight="500"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textSecondary}
                                    >
                                        {getRowLabel(layer)}
                                    </text>
                                {/each}
                            </svg>
                        </div>

                        <!-- Main content area -->
                        <div class="viz-content">
                            <!-- svelte-ignore a11y_no_static_element_interactions -->
                            <svg width={layout.width} height={layout.height}>
                                <!-- Column backgrounds and headers -->
                                {#each intervention.result.input_tokens as token, pos (pos)}
                                    {@const x = layout.seqXStarts[pos]}
                                    {@const w = layout.seqWidths[pos]}
                                    {@const cx = x + w / 2}

                                    <!-- Column header -->
                                    <rect
                                        x={x}
                                        y={0}
                                        width={w}
                                        height={LOGITS_HEADER_HEIGHT}
                                        fill="var(--bg-elevated)"
                                        stroke="var(--border-default)"
                                        stroke-width="1"
                                    />
                                    <text
                                        x={cx}
                                        y={16}
                                        text-anchor="middle"
                                        font-size="9"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textMuted}
                                    >[{pos}]</text>
                                    <text
                                        x={cx}
                                        y={32}
                                        text-anchor="middle"
                                        font-size="11"
                                        font-weight="500"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textPrimary}
                                        style="white-space: pre"
                                    >"{token}"</text>

                                    <!-- Prediction cells -->
                                    {#each Array(MAX_PREDICTIONS) as _, rank (rank)}
                                        {@const pred = intervention.result.predictions_per_position[pos][rank]}
                                        {@const cellY = LOGITS_HEADER_HEIGHT + rank * LOGITS_ROW_HEIGHT}
                                        <rect
                                            x={x}
                                            y={cellY}
                                            width={w}
                                            height={LOGITS_ROW_HEIGHT}
                                            fill={pred ? getProbBgColor(pred.prob) : "var(--bg-inset)"}
                                            stroke="var(--border-subtle)"
                                            stroke-width="1"
                                        />
                                        {#if pred}
                                            <text
                                                x={cx}
                                                y={cellY + 11}
                                                text-anchor="middle"
                                                font-size="10"
                                                font-weight="500"
                                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                                fill={getProbTextColor(pred.prob)}
                                                style="white-space: pre"
                                            >"{pred.token}"</text>
                                            <text
                                                x={cx}
                                                y={cellY + 23}
                                                text-anchor="middle"
                                                font-size="9"
                                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                                fill={getProbTextColor(pred.prob)}
                                                opacity="0.8"
                                            >{formatProb(pred.prob)}</text>
                                        {:else}
                                            <text
                                                x={cx}
                                                y={cellY + LOGITS_ROW_HEIGHT / 2}
                                                text-anchor="middle"
                                                dominant-baseline="middle"
                                                font-size="10"
                                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                                fill={colors.textMuted}
                                            >-</text>
                                        {/if}
                                    {/each}

                                    <!-- Connector line from logits to graph -->
                                    <line
                                        x1={cx}
                                        y1={layout.logitsHeight}
                                        x2={cx}
                                        y2={layout.graphTopY - 10}
                                        stroke="var(--border-subtle)"
                                        stroke-width="1"
                                        stroke-dasharray="2,2"
                                    />

                                    <!-- Token label at bottom of graph area -->
                                    <text
                                        x={cx}
                                        y={layout.height - 20}
                                        text-anchor="middle"
                                        font-size="10"
                                        font-weight="500"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textPrimary}
                                        style="white-space: pre"
                                    >{token}</text>
                                    <text
                                        x={cx}
                                        y={layout.height - 8}
                                        text-anchor="middle"
                                        font-size="9"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textMuted}
                                    >[{pos}]</text>
                                {/each}

                                <!-- Graph section background -->
                                <rect
                                    x={MARGIN.left - 4}
                                    y={layout.graphTopY - 8}
                                    width={layout.width - MARGIN.left - MARGIN.right + 8}
                                    height={layout.height - layout.graphTopY - 30}
                                    fill="var(--bg-inset)"
                                    stroke="var(--border-default)"
                                    stroke-width="1"
                                    rx="2"
                                />

                                <!-- Nodes -->
                                <g class="nodes-layer">
                                    {#each intervention.nodes as node (`${node.layer}:${node.seqIdx}:${node.cIdx}`)}
                                        {@const pos = layout.nodePositions[`${node.layer}:${node.seqIdx}:${node.cIdx}`]}
                                        {#if pos}
                                            <g
                                                class="node-group"
                                                onmouseenter={(e) => handleNodeMouseEnter(e, node.layer, node.seqIdx, node.cIdx)}
                                                onmouseleave={handleNodeMouseLeave}
                                            >
                                                <rect
                                                    x={pos.x - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                                    y={pos.y - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                                    width={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                                    height={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                                    fill="transparent"
                                                />
                                                <rect
                                                    class="node"
                                                    class:highlighted={hoveredNode?.layer === node.layer && hoveredNode?.cIdx === node.cIdx}
                                                    x={pos.x - COMPONENT_SIZE / 2}
                                                    y={pos.y - COMPONENT_SIZE / 2}
                                                    width={COMPONENT_SIZE}
                                                    height={COMPONENT_SIZE}
                                                    fill={colors.nodeDefault}
                                                    rx="1"
                                                    opacity="0.8"
                                                />
                                            </g>
                                        {/if}
                                    {/each}
                                </g>
                            </svg>
                        </div>
                    </div>
                </div>
            {/each}
        </div>
    {/if}

    <!-- Node tooltip -->
    {#if hoveredNode}
        {@const summary = activationContextsSummary?.[hoveredNode.layer]?.find(
            (s) => s.subcomponent_idx === hoveredNode?.cIdx
        )}
        {@const detail = componentDetailsCache[`${hoveredNode.layer}:${hoveredNode.cIdx}`]}
        {@const isLoading = componentDetailsLoading[`${hoveredNode.layer}:${hoveredNode.cIdx}`] ?? false}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
            class="node-tooltip"
            style="left: {tooltipPos.x}px; top: {tooltipPos.y}px;"
            onmouseenter={() => (isHoveringTooltip = true)}
            onmouseleave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
        >
            <h3>{hoveredNode.layer}:{hoveredNode.seqIdx}:{hoveredNode.cIdx}</h3>
            <ComponentDetailCard
                layer={hoveredNode.layer}
                cIdx={hoveredNode.cIdx}
                seqIdx={hoveredNode.seqIdx}
                {detail}
                {isLoading}
                outputProbs={{}}
                {summary}
                compact
            />
        </div>
    {/if}
</div>

<style>
    .interventions-view {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .empty-state {
        display: flex;
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        text-align: center;
        padding: var(--space-4);
        font-family: var(--font-sans);
        background: var(--bg-surface);
    }

    .empty-state p {
        margin: var(--space-1) 0;
    }

    .empty-state .hint {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .interventions-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-2) 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
    }

    .clear-btn {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }

    .clear-btn:hover {
        background: var(--status-negative);
        color: white;
        border-color: var(--status-negative);
    }

    .interventions-list {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .intervention-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .intervention-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-3);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .intervention-title {
        font-weight: 600;
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .intervention-time {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    /* Unified visualization */
    .unified-viz-wrapper {
        display: flex;
        border: 1px solid var(--border-default);
        background: var(--bg-surface);
        overflow: hidden;
    }

    .labels-column {
        flex-shrink: 0;
        background: var(--bg-surface);
        border-right: 1px solid var(--border-default);
    }

    .viz-content {
        overflow-x: auto;
        flex: 1;
        background: var(--bg-surface);
    }

    .node-group {
        cursor: pointer;
    }

    .node {
        transition: opacity 0.1s;
    }

    .node.highlighted {
        stroke: var(--accent-primary);
        stroke-width: 2px;
        filter: brightness(1.2);
        opacity: 1 !important;
    }

    .node-tooltip {
        position: fixed;
        padding: var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        z-index: 1000;
        pointer-events: auto;
        font-family: var(--font-mono);
        max-width: 400px;
        max-height: 400px;
        overflow-y: auto;
    }

    .node-tooltip h3 {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-base);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        font-weight: 600;
        letter-spacing: 0.02em;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: var(--space-2);
    }
</style>
