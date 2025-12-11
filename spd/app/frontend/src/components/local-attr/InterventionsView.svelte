<script lang="ts">
    import { SvelteSet } from "svelte/reactivity";
    import type { StoredGraph } from "./types";
    import {
        isInterventableNode,
        type ActivationContextsSummary,
        type ComponentDetail,
        type LayerInfo,
        type NodePosition,
    } from "../../lib/localAttributionsTypes";
    import { colors, getEdgeColor, getOutputHeaderColor } from "../../lib/colors";
    import { lerp, calcTooltipPos } from "./graphUtils";
    import NodeTooltip from "./NodeTooltip.svelte";

    // Layout constants
    const COMPONENT_SIZE = 8;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;
    const ROW_ORDER = ["wte", "qkv", "o_proj", "c_fc", "down_proj", "output"];
    const QKV_SUBTYPES = ["q_proj", "k_proj", "v_proj"];

    // Logits display constants
    const MAX_PREDICTIONS = 5;

    type Props = {
        graph: StoredGraph;
        tokens: string[];
        initialTopK: number;
        activationContextsSummary: ActivationContextsSummary | null;
        componentDetailsCache: Record<string, ComponentDetail>;
        componentDetailsLoading: Record<string, boolean>;
        runningIntervention: boolean;
        onLoadComponentDetail: (layer: string, cIdx: number) => void;
        onSelectionChange: (selection: SvelteSet<string>) => void;
        onRunIntervention: () => void;
        onSelectRun: (runId: number) => void;
        onDeleteRun: (runId: number) => void;
    };

    let {
        graph,
        tokens,
        initialTopK,
        activationContextsSummary,
        componentDetailsCache,
        componentDetailsLoading,
        runningIntervention,
        onLoadComponentDetail,
        onSelectionChange,
        onRunIntervention,
        onSelectRun,
        onDeleteRun,
    }: Props = $props();

    // Local topK state for the composer
    let topK = $state(initialTopK);

    // Composer state
    let componentGap = $state(4);
    let layerGap = $state(30);

    // Hover state for composer
    let hoveredNode = $state<{ layer: string; seqIdx: number; cIdx: number } | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state({ x: 0, y: 0 });
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    // Drag-to-select state
    let isDragging = $state(false);
    let dragStart = $state<{ x: number; y: number } | null>(null);
    let dragCurrent = $state<{ x: number; y: number } | null>(null);
    let svgElement: SVGSVGElement | null = null;

    // Parse layer name
    function parseLayer(name: string): LayerInfo {
        if (name === "wte") return { name, block: -1, type: "embed", subtype: "wte" };
        if (name === "output") return { name, block: Infinity, type: "output", subtype: "output" };
        const m = name.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
        if (!m) throw new Error(`parseLayer: unrecognized layer name: ${name}`);
        return { name, block: +m[1], type: m[2] as "attn" | "mlp", subtype: m[3] };
    }

    function getRowKey(layer: string): string {
        const info = parseLayer(layer);
        if (QKV_SUBTYPES.includes(info.subtype)) return `h.${info.block}.qkv`;
        return layer;
    }

    // All nodes from the graph (for rendering)
    const allNodes = $derived(new SvelteSet(Object.keys(graph.data.nodeCiVals)));

    // Interventable nodes only (for selection)
    const interventableNodes = $derived.by(() => {
        const nodes = new SvelteSet<string>();
        for (const nodeKey of allNodes) {
            if (isInterventableNode(nodeKey)) nodes.add(nodeKey);
        }
        return nodes;
    });

    // Filter edges for rendering (topK by magnitude)
    const filteredEdges = $derived.by(() => {
        const edgesCopy = [...graph.data.edges];
        const sortedEdges = edgesCopy.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
        return sortedEdges.slice(0, topK);
    });

    // Compute layout for composer
    const layout = $derived.by(() => {
        const nodesPerLayerSeq: Record<string, number[]> = {};
        const allLayers = new SvelteSet<string>();
        const allRows = new SvelteSet<string>();

        for (const nodeKey of allNodes) {
            const [layer, seqIdx, cIdx] = nodeKey.split(":");
            allLayers.add(layer);
            allRows.add(getRowKey(layer));
            const key = `${layer}:${seqIdx}`;
            if (!nodesPerLayerSeq[key]) nodesPerLayerSeq[key] = [];
            nodesPerLayerSeq[key].push(+cIdx);
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
            return ROW_ORDER.indexOf(infoA.subtype) - ROW_ORDER.indexOf(infoB.subtype);
        });

        const numTokens = tokens.length;

        // Calculate column widths
        const maxComponentsPerSeq = Array.from({ length: numTokens }, (_, seqIdx) => {
            let maxAtSeq = 1;
            for (const row of rows) {
                if (row.endsWith(".qkv")) {
                    const blockMatch = row.match(/h\.(\d+)/);
                    if (blockMatch) {
                        const block = blockMatch[1];
                        let totalQkv = 0;
                        for (const subtype of QKV_SUBTYPES) {
                            const layer = `h.${block}.attn.${subtype}`;
                            totalQkv += (nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? []).length;
                        }
                        totalQkv += 2;
                        maxAtSeq = Math.max(maxAtSeq, totalQkv);
                    }
                } else {
                    for (const layer of allLayers) {
                        if (getRowKey(layer) === row) {
                            maxAtSeq = Math.max(maxAtSeq, (nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? []).length);
                        }
                    }
                }
            }
            return maxAtSeq;
        });

        const COL_PADDING = 12;
        const MIN_COL_WIDTH = 60;
        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + componentGap) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left + LABEL_WIDTH];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Y positions
        const rowYPositions: Record<string, number> = {};
        let currentY = MARGIN.top;
        for (const row of rows.slice().reverse()) {
            rowYPositions[row] = currentY;
            currentY += COMPONENT_SIZE + layerGap;
        }

        const layerYPositions: Record<string, number> = {};
        for (const layer of allLayers) {
            layerYPositions[layer] = rowYPositions[getRowKey(layer)];
        }

        // Position nodes
        const nodePositions: Record<string, NodePosition> = {};
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
                        baseX +=
                            (nodesPerLayerSeq[`${prevLayer}:${seqIdx}`]?.length ?? 0) * (COMPONENT_SIZE + componentGap);
                        baseX += COMPONENT_SIZE + componentGap;
                    }
                }

                layerNodes.sort((a, b) => a - b);
                layerNodes.forEach((cIdx, i) => {
                    nodePositions[`${layer}:${seqIdx}:${cIdx}`] = {
                        x: baseX + i * (COMPONENT_SIZE + componentGap) + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                });
            }
        }

        const totalSeqWidth = seqXStarts[seqXStarts.length - 1] + seqWidths[seqWidths.length - 1];
        const width = totalSeqWidth + MARGIN.right;
        const maxY = rows.length > 0 ? Math.max(...Object.values(layerYPositions)) + COMPONENT_SIZE : MARGIN.top;
        const height = maxY + MARGIN.bottom + 40;

        return {
            nodePositions,
            layerYPositions,
            seqWidths,
            seqXStarts,
            width,
            height,
            nodesPerLayerSeq,
            allLayers,
            rows,
        };
    });

    // Derived values
    const maxAbsAttr = $derived(graph.data.maxAbsAttr || 1);
    const selectedCount = $derived(graph.composerSelection.size);
    const interventableCount = $derived(interventableNodes.size);

    // Selection helpers
    function isNodeSelected(nodeKey: string): boolean {
        return graph.composerSelection.has(nodeKey);
    }

    function toggleNode(nodeKey: string) {
        if (!isInterventableNode(nodeKey)) return; // Can't toggle non-interventable nodes
        const newSelection = new SvelteSet(graph.composerSelection);
        if (newSelection.has(nodeKey)) {
            newSelection.delete(nodeKey);
        } else {
            newSelection.add(nodeKey);
        }
        onSelectionChange(newSelection);
    }

    function selectAll() {
        onSelectionChange(new SvelteSet(interventableNodes));
    }

    function clearSelection() {
        onSelectionChange(new SvelteSet());
    }

    // Hover handlers
    function handleNodeMouseEnter(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        if (isDragging) return;
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
            if (!isHoveringTooltip) hoveredNode = null;
            hoverTimeout = null;
        }, 50);
    }

    function handleNodeClick(nodeKey: string) {
        toggleNode(nodeKey);
        hoveredNode = null;
    }

    // Drag-to-select handlers
    function getSvgPoint(event: MouseEvent): { x: number; y: number } | null {
        if (!svgElement) return null;
        const container = svgElement.parentElement!;
        const rect = container.getBoundingClientRect();
        return {
            x: event.clientX - rect.left + container.scrollLeft,
            y: event.clientY - rect.top + container.scrollTop,
        };
    }

    function handleSvgMouseDown(event: MouseEvent) {
        // Only start drag on left mouse button and not on a node
        if (event.button !== 0) return;
        const target = event.target as Element;
        if (target.closest(".node-group")) return;

        event.preventDefault(); // Prevent text selection while dragging

        const point = getSvgPoint(event);
        if (!point) return;

        hoveredNode = null; // Clear tooltip when starting drag
        isDragging = true;
        dragStart = point;
        dragCurrent = point;
    }

    function handleSvgMouseMove(event: MouseEvent) {
        if (!isDragging) return;
        dragCurrent = getSvgPoint(event);
    }

    function handleSvgMouseUp() {
        if (!isDragging || !dragStart || !dragCurrent) {
            isDragging = false;
            dragStart = null;
            dragCurrent = null;
            return;
        }

        // Calculate selection rectangle bounds
        const minX = Math.min(dragStart.x, dragCurrent.x);
        const maxX = Math.max(dragStart.x, dragCurrent.x);
        const minY = Math.min(dragStart.y, dragCurrent.y);
        const maxY = Math.max(dragStart.y, dragCurrent.y);

        // Only select if drag was meaningful (more than a few pixels)
        const dragDistance = Math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2);
        if (dragDistance > 5) {
            // Find nodes within the selection rectangle
            const nodesToToggle: string[] = [];
            for (const nodeKey of interventableNodes) {
                const pos = layout.nodePositions[nodeKey];
                if (!pos) continue;

                // Check if node center is within selection rect
                if (pos.x >= minX && pos.x <= maxX && pos.y >= minY && pos.y <= maxY) {
                    nodesToToggle.push(nodeKey);
                }
            }

            // Toggle selection for nodes in rect
            if (nodesToToggle.length > 0) {
                const newSelection = new SvelteSet(graph.composerSelection);
                for (const nodeKey of nodesToToggle) {
                    if (newSelection.has(nodeKey)) {
                        newSelection.delete(nodeKey);
                    } else {
                        newSelection.add(nodeKey);
                    }
                }
                onSelectionChange(newSelection);
            }
        }

        isDragging = false;
        dragStart = null;
        dragCurrent = null;
    }

    // Derived selection rectangle for rendering
    const selectionRect = $derived.by(() => {
        if (!isDragging || !dragStart || !dragCurrent) return null;
        return {
            x: Math.min(dragStart.x, dragCurrent.x),
            y: Math.min(dragStart.y, dragCurrent.y),
            width: Math.abs(dragCurrent.x - dragStart.x),
            height: Math.abs(dragCurrent.y - dragStart.y),
        };
    });

    // Edge rendering
    function getEdgePath(src: string, tgt: string): string {
        const srcPos = layout.nodePositions[src];
        const tgtPos = layout.nodePositions[tgt];
        if (!srcPos || !tgtPos) return "";

        const midY = (srcPos.y + tgtPos.y) / 2;
        return `M ${srcPos.x} ${srcPos.y} C ${srcPos.x} ${midY}, ${tgtPos.x} ${midY}, ${tgtPos.x} ${tgtPos.y}`;
    }

    function getEdgeOpacity(val: number): number {
        const normalized = Math.abs(val) / maxAbsAttr;
        return lerp(0.1, 0.8, Math.sqrt(normalized));
    }

    function getEdgeWidth(val: number): number {
        const normalized = Math.abs(val) / maxAbsAttr;
        return lerp(0.5, 3, Math.sqrt(normalized));
    }

    // Run history helpers
    function formatTime(timestamp: string): string {
        return new Date(timestamp).toLocaleTimeString();
    }

    function formatProb(prob: number): string {
        if (prob >= 0.01) return (prob * 100).toFixed(1) + "%";
        return (prob * 100).toExponential(1) + "%";
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
    <!-- Composer Panel (Left) -->
    <div class="composer-panel">
        <div class="composer-header">
            <span class="title">Composer</span>
            <span class="node-count">{selectedCount} / {interventableCount} nodes selected</span>
        </div>

        <div class="composer-controls">
            <div class="topk-control">
                <label for="topk-slider">Show Top K Edges:</label>
                <input id="topk-slider" type="range" min="10" max="2000" step="10" bind:value={topK} />
                <span class="topk-value">{topK}</span>
            </div>
            <div class="button-group">
                <button onclick={selectAll}>Select All</button>
                <button onclick={clearSelection}>Clear All</button>
                <button
                    class="run-btn"
                    onclick={onRunIntervention}
                    disabled={selectedCount === 0 || runningIntervention}
                >
                    {runningIntervention ? "Running..." : "Run"}
                </button>
            </div>
        </div>

        <div class="composer-hint">Click to toggle, drag to select multiple</div>

        <div class="composer-graph">
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <svg
                bind:this={svgElement}
                style="min-width: {layout.width}px; min-height: {layout.height}px; width: 100%; height: 100%;"
                onmousedown={handleSvgMouseDown}
                onmousemove={handleSvgMouseMove}
                onmouseup={handleSvgMouseUp}
                onmouseleave={handleSvgMouseUp}
            >
                <!-- Token headers -->
                {#each tokens as token, pos (pos)}
                    {@const x = layout.seqXStarts[pos]}
                    {@const w = layout.seqWidths[pos]}
                    {@const cx = x + w / 2}
                    <text
                        x={cx}
                        y={MARGIN.top - 30}
                        text-anchor="middle"
                        font-size="10"
                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                        fill={colors.textPrimary}
                        style="white-space: pre">"{token}"</text
                    >
                    <text
                        x={cx}
                        y={MARGIN.top - 18}
                        text-anchor="middle"
                        font-size="9"
                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                        fill={colors.textMuted}>[{pos}]</text
                    >
                {/each}

                <!-- Layer labels -->
                {#each Object.entries(layout.layerYPositions) as [layer, y] (layer)}
                    <text
                        x={MARGIN.left + LABEL_WIDTH - 8}
                        y={y + COMPONENT_SIZE / 2}
                        text-anchor="end"
                        dominant-baseline="middle"
                        font-size="10"
                        font-weight="500"
                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                        fill={colors.textSecondary}>{getRowLabel(layer)}</text
                    >
                {/each}

                <!-- Edges -->
                <g class="edges-layer" opacity="0.6">
                    {#each filteredEdges as edge (`${edge.src}-${edge.tgt}`)}
                        {@const path = getEdgePath(edge.src, edge.tgt)}
                        {#if path}
                            <path
                                d={path}
                                stroke={getEdgeColor(edge.val)}
                                stroke-width={getEdgeWidth(edge.val)}
                                fill="none"
                                opacity={getEdgeOpacity(edge.val)}
                            />
                        {/if}
                    {/each}
                </g>

                <!-- Nodes -->
                <g class="nodes-layer">
                    {#each allNodes as nodeKey (nodeKey)}
                        {@const pos = layout.nodePositions[nodeKey]}
                        {@const [layer, seqIdx, cIdx] = nodeKey.split(":")}
                        {@const interventable = isInterventableNode(nodeKey)}
                        {@const selected = interventable && isNodeSelected(nodeKey)}
                        {@const isOutput = layer === "output"}
                        {@const outputEntry = isOutput ? graph.data.outputProbs[`${seqIdx}:${cIdx}`] : null}
                        {#if pos}
                            <!-- svelte-ignore a11y_click_events_have_key_events -->
                            <!-- svelte-ignore a11y_no_static_element_interactions -->
                            <g
                                class="node-group"
                                class:selected
                                class:non-interventable={!interventable}
                                onmouseenter={(e) => handleNodeMouseEnter(e, layer, +seqIdx, +cIdx)}
                                onmouseleave={handleNodeMouseLeave}
                                onclick={() => handleNodeClick(nodeKey)}
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
                                    x={pos.x - COMPONENT_SIZE / 2}
                                    y={pos.y - COMPONENT_SIZE / 2}
                                    width={COMPONENT_SIZE}
                                    height={COMPONENT_SIZE}
                                    fill={!interventable
                                        ? colors.textMuted
                                        : selected
                                          ? colors.accent
                                          : colors.nodeDefault}
                                    stroke={selected ? colors.accent : "none"}
                                    stroke-width={selected ? 2 : 0}
                                    rx="1"
                                    opacity={!interventable ? 0.3 : selected ? 1 : 0.4}
                                />
                                {#if isOutput && outputEntry}
                                    <text
                                        x={pos.x}
                                        y={pos.y + COMPONENT_SIZE + 10}
                                        text-anchor="middle"
                                        font-size="8"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textMuted}>"{outputEntry.token}"</text
                                    >
                                {/if}
                            </g>
                        {/if}
                    {/each}
                </g>

                <!-- Selection rectangle -->
                {#if selectionRect}
                    <rect
                        class="selection-rect"
                        x={selectionRect.x}
                        y={selectionRect.y}
                        width={selectionRect.width}
                        height={selectionRect.height}
                        fill="rgba(99, 102, 241, 0.1)"
                        stroke={colors.accent}
                        stroke-width="1"
                        stroke-dasharray="4 2"
                    />
                {/if}
            </svg>
        </div>
    </div>

    <!-- Run History Panel (Right) -->
    <div class="history-panel">
        <div class="history-header">
            <span class="title">Run History</span>
            <span class="run-count">{graph.interventionRuns.length} runs</span>
        </div>

        {#if graph.interventionRuns.length === 0}
            <div class="empty-history">
                <p>No runs yet</p>
                <p class="hint">Select nodes and click Run</p>
            </div>
        {:else}
            <div class="runs-list">
                {#each graph.interventionRuns.slice().reverse() as run (run.id)}
                    {@const isActive = graph.activeRunId === run.id}
                    <div
                        class="run-card"
                        class:active={isActive}
                        role="button"
                        tabindex="0"
                        onclick={() => onSelectRun(run.id)}
                        onkeydown={(e) => e.key === "Enter" && onSelectRun(run.id)}
                    >
                        <div class="run-header">
                            <span class="run-time">{formatTime(run.created_at)}</span>
                            <span class="run-nodes">{run.selected_nodes.length} nodes</span>
                            <button
                                class="delete-btn"
                                onclick={(e) => {
                                    e.stopPropagation();
                                    onDeleteRun(run.id);
                                }}>✕</button
                            >
                        </div>

                        <!-- Mini logits table -->
                        <div class="logits-mini">
                            <table>
                                <thead>
                                    <tr>
                                        {#each run.result.input_tokens as token, idx (idx)}
                                            <th title={token}>
                                                <span class="token-text">"{token}"</span>
                                            </th>
                                        {/each}
                                    </tr>
                                </thead>
                                <tbody>
                                    {#each Array(Math.min(3, MAX_PREDICTIONS)) as _, rank (rank)}
                                        <tr>
                                            {#each run.result.predictions_per_position as preds, idx (idx)}
                                                {@const pred = preds[rank]}
                                                <td
                                                    class:has-pred={!!pred}
                                                    style={pred ? `background: ${getOutputHeaderColor(pred.prob)}` : ""}
                                                >
                                                    {#if pred}
                                                        <span class="pred-token">"{pred.token}"</span>
                                                        <span class="pred-prob">{formatProb(pred.prob)}</span>
                                                    {:else}
                                                        -
                                                    {/if}
                                                </td>
                                            {/each}
                                        </tr>
                                    {/each}
                                </tbody>
                            </table>
                        </div>
                    </div>
                {/each}
            </div>
        {/if}
    </div>

    <!-- Node tooltip -->
    {#if hoveredNode}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            {activationContextsSummary}
            {componentDetailsCache}
            {componentDetailsLoading}
            outputProbs={graph.data.outputProbs}
            nodeCiVals={graph.data.nodeCiVals}
            {tokens}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
        />
    {/if}
</div>

<style>
    .interventions-view {
        display: flex;
        flex: 1;
        min-height: 0;
        gap: var(--space-4);
    }

    /* Composer Panel */
    .composer-panel {
        flex: 2;
        display: flex;
        flex-direction: column;
        min-width: 0;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .composer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .composer-header .title {
        font-weight: 600;
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .node-count {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .composer-controls {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        margin-bottom: var(--space-2);
    }

    .topk-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .topk-control label {
        white-space: nowrap;
    }

    .topk-control input[type="range"] {
        flex: 1;
        min-width: 100px;
        max-width: 200px;
    }

    .topk-value {
        min-width: 40px;
        text-align: right;
        color: var(--text-primary);
    }

    .button-group {
        display: flex;
        gap: var(--space-2);
    }

    .button-group button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }

    .button-group button:hover:not(:disabled) {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .run-btn {
        background: var(--accent-primary) !important;
        color: white !important;
        border-color: var(--accent-primary) !important;
    }

    .run-btn:hover:not(:disabled) {
        background: var(--accent-primary-dim) !important;
    }

    .run-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .composer-hint {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        margin-bottom: var(--space-2);
    }

    .composer-graph {
        flex: 1;
        overflow: auto;
        background: var(--bg-inset);
        border: 1px solid var(--border-subtle);
    }

    .composer-graph svg {
        cursor: crosshair;
    }

    .node-group {
        cursor: pointer;
    }

    .node-group .node {
        transition:
            opacity 0.1s,
            fill 0.1s;
    }

    .node-group:hover .node {
        opacity: 1 !important;
        filter: brightness(1.2);
    }

    .node-group.non-interventable {
        cursor: default;
    }

    .node-group.non-interventable:hover .node {
        filter: none;
    }

    /* History Panel */
    .history-panel {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 300px;
        max-width: 400px;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .history-header .title {
        font-weight: 600;
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .run-count {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .empty-history {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        text-align: center;
    }

    .empty-history p {
        margin: var(--space-1) 0;
    }

    .empty-history .hint {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .runs-list {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .run-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        padding: var(--space-2);
        cursor: pointer;
        transition: border-color 0.1s;
    }

    .run-card:hover {
        border-color: var(--border-strong);
    }

    .run-card.active {
        border-color: var(--accent-primary);
        background: var(--bg-inset);
    }

    .run-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin-bottom: var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .run-time {
        color: var(--text-secondary);
    }

    .run-nodes {
        color: var(--text-muted);
        margin-left: auto;
    }

    .delete-btn {
        padding: 2px 6px;
        background: transparent;
        border: none;
        color: var(--text-muted);
        font-size: var(--text-xs);
    }

    .delete-btn:hover {
        color: var(--status-negative);
    }

    /* Mini logits table */
    .logits-mini {
        overflow-x: auto;
    }

    .logits-mini table {
        width: 100%;
        border-collapse: collapse;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .logits-mini th,
    .logits-mini td {
        padding: 2px 4px;
        text-align: center;
        border: 1px solid var(--border-subtle);
        max-width: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .logits-mini th {
        background: var(--bg-surface);
        color: var(--text-secondary);
    }

    .logits-mini .token-text {
        font-size: 9px;
    }

    .logits-mini td {
        background: var(--bg-inset);
        color: var(--text-muted);
    }

    .logits-mini td.has-pred {
        background: var(--bg-surface);
    }

    .pred-token {
        display: block;
        color: var(--text-primary);
    }

    .pred-prob {
        display: block;
        font-size: 8px;
        color: var(--text-muted);
    }
</style>
