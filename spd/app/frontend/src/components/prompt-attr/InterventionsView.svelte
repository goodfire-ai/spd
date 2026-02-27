<script lang="ts">
    import { getContext } from "svelte";
    import { SvelteSet } from "svelte/reactivity";
    import { colors, getEdgeColor, getOutputHeaderColor, rgbaToCss } from "../../lib/colors";
    import type { Loadable } from "../../lib/index";
    import type { NormalizeType } from "../../lib/api";
    import { isInterventableNode, type NodePosition } from "../../lib/promptAttributionsTypes";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";
    import {
        parseLayer,
        getRowKey as _getRowKey,
        getRowLabel as _getRowLabel,
        sortRows,
        getGroupProjections,
        buildLayerAddress,
    } from "../../lib/graphLayout";

    const runState = getContext<RunContext>(RUN_KEY);
    import {
        calcTooltipPos,
        computeClusterSpans,
        computeComponentOffsets,
        lerp,
        sortComponentsByCluster,
        sortComponentsByImportance,
        type ClusterSpan,
        type TooltipPos,
    } from "./graphUtils";
    import NodeTooltip from "./NodeTooltip.svelte";
    import TokenDropdown from "./TokenDropdown.svelte";
    import type { StoredGraph } from "./types";
    import ViewControls from "./ViewControls.svelte";
    import { useZoomPan } from "../../lib/useZoomPan.svelte";
    import ZoomControls from "../../lib/ZoomControls.svelte";

    // Layout constants
    const COMPONENT_SIZE = 6;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;
    const CLUSTER_BAR_HEIGHT = 3;
    const CLUSTER_BAR_GAP = 2;
    const LAYER_X_OFFSET = 3; // Horizontal offset per layer to avoid edge overlap

    // Logits display constants
    const MAX_PREDICTIONS = 5;

    import type { ForkedInterventionRunSummary } from "../../lib/interventionTypes";

    type Props = {
        graph: StoredGraph;
        composerSelection: Set<string>;
        activeRunId: number | null;
        tokens: string[];
        tokenIds: number[];
        // View settings (shared with main graph)
        topK: number;
        componentGap: number;
        layerGap: number;
        normalizeEdges: NormalizeType;
        ciThreshold: Loadable<number>;
        hideUnpinnedEdges: boolean;
        hideNodeCard: boolean;
        onTopKChange: (value: number) => void;
        onComponentGapChange: (value: number) => void;
        onLayerGapChange: (value: number) => void;
        onNormalizeChange: (value: NormalizeType) => void;
        onCiThresholdChange: (value: number) => void;
        onHideUnpinnedEdgesChange: (value: boolean) => void;
        onHideNodeCardChange: (value: boolean) => void;
        // Other props
        runningIntervention: boolean;
        generatingSubgraph: boolean;
        onSelectionChange: (selection: Set<string>) => void;
        onRunIntervention: () => void;
        onSelectRun: (runId: number) => void;
        onDeleteRun: (runId: number) => void;
        onForkRun: (runId: number, tokenReplacements: [number, number][]) => Promise<ForkedInterventionRunSummary>;
        onDeleteFork: (forkId: number) => void;
        onGenerateGraphFromSelection: () => void;
    };

    let {
        graph,
        composerSelection,
        activeRunId,
        tokens,
        tokenIds,
        topK,
        componentGap,
        layerGap,
        normalizeEdges,
        ciThreshold,
        hideUnpinnedEdges,
        hideNodeCard,
        onTopKChange,
        onComponentGapChange,
        onLayerGapChange,
        onNormalizeChange,
        onCiThresholdChange,
        onHideUnpinnedEdgesChange,
        onHideNodeCardChange,
        runningIntervention,
        generatingSubgraph,
        onSelectionChange,
        onRunIntervention,
        onSelectRun,
        onDeleteRun,
        onForkRun,
        onDeleteFork,
        onGenerateGraphFromSelection,
    }: Props = $props();

    // Track newly-added run for flash animation
    let knownRunIds = new Set(graph.interventionRuns.map((r) => r.id));
    let flashRunId = $state<number | null>(null);

    $effect(() => {
        const currentIds = new Set(graph.interventionRuns.map((r) => r.id));
        for (const id of currentIds) {
            if (!knownRunIds.has(id)) {
                flashRunId = id;
                setTimeout(() => (flashRunId = null), 1500);
                break;
            }
        }
        knownRunIds = currentIds;
    });

    // Fork modal state
    // Per-position state: { value: display string, tokenId: selected token ID or null }
    type ForkSlotState = { value: string; tokenId: number | null };
    let forkingRunId = $state<number | null>(null);
    let forkSlotStates = $state<ForkSlotState[]>([]);
    let forkingInProgress = $state(false);

    function openForkModal(runId: number) {
        forkingRunId = runId;
        // Initialize each slot with the original token
        forkSlotStates = tokens.map((tok, idx) => ({
            value: tok,
            tokenId: tokenIds[idx],
        }));
    }

    function closeForkModal() {
        forkingRunId = null;
        forkSlotStates = [];
    }

    function handleForkSlotSelect(seqPos: number, tokenId: number | null, tokenString: string) {
        forkSlotStates = forkSlotStates.map((slot, idx) => (idx === seqPos ? { value: tokenString, tokenId } : slot));
    }

    function resetForkSlot(seqPos: number) {
        forkSlotStates = forkSlotStates.map((slot, idx) =>
            idx === seqPos ? { value: tokens[idx], tokenId: tokenIds[idx] } : slot,
        );
    }

    // Computed: which positions have valid replacements (different from original)
    const forkReplacements = $derived.by(() => {
        const replacements: [number, number][] = [];
        for (let i = 0; i < forkSlotStates.length; i++) {
            const slot = forkSlotStates[i];
            if (slot.tokenId !== null && slot.tokenId !== tokenIds[i]) {
                replacements.push([i, slot.tokenId]);
            }
        }
        return replacements;
    });

    async function submitFork() {
        if (forkingRunId === null || forkReplacements.length === 0) return;
        forkingInProgress = true;
        try {
            await onForkRun(forkingRunId, forkReplacements);
            closeForkModal();
        } finally {
            forkingInProgress = false;
        }
    }

    // Hover state for composer
    let hoveredNode = $state<{ layer: string; seqIdx: number; cIdx: number } | null>(null);
    let hoveredBarClusterId = $state<number | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state<TooltipPos>({ left: 0, top: 0 });
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    // Refs
    let graphContainer: HTMLDivElement;

    // Zoom/pan
    const zoom = useZoomPan(() => graphContainer);

    // Get cluster ID of hovered node or bar (for cluster-wide rotation effect)
    const hoveredClusterId = $derived.by(() => {
        if (hoveredBarClusterId !== null) return hoveredBarClusterId;
        if (!hoveredNode) return undefined;
        return runState.getClusterId(hoveredNode.layer, hoveredNode.cIdx);
    });

    // Check if a node is in the same cluster as the hovered node (for cluster rotation effect)
    function isNodeInSameCluster(nodeKey: string): boolean {
        // Only trigger if hovered node has a numeric cluster ID (not singleton/no mapping)
        if (hoveredClusterId === undefined || hoveredClusterId === null) return false;
        const [layer, , cIdxStr] = nodeKey.split(":");
        const cIdx = parseInt(cIdxStr);
        const nodeClusterId = runState.getClusterId(layer, cIdx);
        return nodeClusterId === hoveredClusterId;
    }

    // Check if a node matches the hovered component (same layer:cIdx across any seqIdx)
    function nodeMatchesHoveredComponent(nodeKey: string): boolean {
        if (!hoveredNode) return false;
        const [layer, , cIdxStr] = nodeKey.split(":");
        const cIdx = parseInt(cIdxStr);
        return layer === hoveredNode.layer && cIdx === hoveredNode.cIdx;
    }

    // Drag-to-select state
    let isDragging = $state(false);
    let dragStart = $state<{ x: number; y: number } | null>(null);
    let dragCurrent = $state<{ x: number; y: number } | null>(null);
    let svgElement: SVGSVGElement | null = null;

    function getRowKey(layer: string): string {
        return _getRowKey(layer);
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

    // Filter edges for rendering (topK by magnitude, optionally hide edges not connected to selected nodes).
    // Edges arrive pre-sorted by abs(val) desc from backend, so filter preserves order and we just slice.
    const filteredEdges = $derived.by(() => {
        let edges = graph.data.edges;
        if (hideUnpinnedEdges && composerSelection.size > 0) {
            edges = edges.filter((e) => composerSelection.has(e.src) || composerSelection.has(e.tgt));
        }
        return edges.slice(0, topK);
    });

    // Edge count for ViewControls
    const filteredEdgeCount = $derived(filteredEdges.length);

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
        const rows = sortRows(Array.from(allRows));

        const numTokens = tokens.length;

        // Calculate column widths
        const maxComponentsPerSeq = Array.from({ length: numTokens }, (_, seqIdx) => {
            let maxAtSeq = 1;
            for (const row of rows) {
                let totalInRow = 0;
                for (const layer of allLayers) {
                    if (getRowKey(layer) === row) {
                        totalInRow += (nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? []).length;
                    }
                }
                const rowParts = row.split(".");
                const isGroupedRow = rowParts.length >= 3 && rowParts[2].includes("_");
                if (isGroupedRow) {
                    const groupProjs = getGroupProjections(rowParts[1]);
                    if (groupProjs && groupProjs.length > 1) {
                        totalInRow += groupProjs.length - 1;
                    }
                }
                maxAtSeq = Math.max(maxAtSeq, totalInRow);
            }
            return maxAtSeq;
        });

        const COL_PADDING = 12;
        const MIN_COL_WIDTH = 60;
        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + componentGap) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Assign Y positions (output at top, wte at bottom)
        const rowYPositions: Record<string, number> = {};
        for (let i = 0; i < rows.length; i++) {
            const distanceFromEnd = rows.length - 1 - i;
            rowYPositions[rows[i]] = MARGIN.top + distanceFromEnd * (COMPONENT_SIZE + layerGap);
        }

        // Map each layer to its row's Y position and X offset
        // X offset: output (last row) at center, others alternate +/- based on distance
        const layerYPositions: Record<string, number> = {};
        const layerXOffsets: Record<string, number> = {};
        for (const layer of allLayers) {
            const rowKey = getRowKey(layer);
            layerYPositions[layer] = rowYPositions[rowKey];
            const rowIdx = rows.indexOf(rowKey);
            const distanceFromOutput = rows.length - 1 - rowIdx;
            if (distanceFromOutput === 0 || layer === "embed") {
                layerXOffsets[layer] = 0;
            } else {
                layerXOffsets[layer] = distanceFromOutput % 2 === 1 ? LAYER_X_OFFSET : -LAYER_X_OFFSET;
            }
        }

        // Position nodes and compute cluster spans
        const nodePositions: Record<string, NodePosition> = {};
        const allClusterSpans: ClusterSpan[] = [];
        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const groupProjs = info.sublayer ? getGroupProjections(info.sublayer) : null;
            const isGrouped = groupProjs !== null && info.projection !== null && groupProjs.includes(info.projection);

            for (let seqIdx = 0; seqIdx < numTokens; seqIdx++) {
                const layerNodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!layerNodes) continue;

                let baseX = seqXStarts[seqIdx] + COL_PADDING + layerXOffsets[layer];
                const baseY = layerYPositions[layer];

                if (isGrouped && groupProjs && info.projection) {
                    const projIdx = groupProjs.indexOf(info.projection);
                    for (let i = 0; i < projIdx; i++) {
                        const prevLayer = buildLayerAddress(info.block, info.sublayer, groupProjs[i]);
                        baseX +=
                            (nodesPerLayerSeq[`${prevLayer}:${seqIdx}`]?.length ?? 0) * (COMPONENT_SIZE + componentGap);
                        baseX += COMPONENT_SIZE + componentGap;
                    }
                }

                // Output nodes always sort by probability; internal nodes sort by cluster if mapping loaded, else by CI
                const sorted =
                    layer === "output" || !runState.clusterMapping
                        ? sortComponentsByImportance(
                              layerNodes,
                              layer,
                              seqIdx,
                              graph.data.nodeCiVals,
                              graph.data.outputProbs,
                          )
                        : sortComponentsByCluster(
                              layerNodes,
                              layer,
                              seqIdx,
                              graph.data.nodeCiVals,
                              runState.getClusterId,
                          );
                const offsets = computeComponentOffsets(sorted, COMPONENT_SIZE, componentGap);
                for (const cIdx of layerNodes) {
                    nodePositions[`${layer}:${seqIdx}:${cIdx}`] = {
                        x: baseX + offsets[cIdx] + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                }

                // Compute cluster spans for this layer/seqIdx (skip output layer)
                if (layer !== "output" && runState.clusterMapping) {
                    const spans = computeClusterSpans(
                        sorted,
                        layer,
                        seqIdx,
                        baseX,
                        baseY,
                        COMPONENT_SIZE,
                        offsets,
                        runState.getClusterId,
                    );
                    allClusterSpans.push(...spans);
                }
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
            clusterSpans: allClusterSpans,
        };
    });

    // Derived values
    const maxAbsAttr = $derived(graph.data.maxAbsAttr || 1);
    const selectedCount = $derived(composerSelection.size);
    const interventableCount = $derived(interventableNodes.size);

    // Selection helpers
    function isNodeSelected(nodeKey: string): boolean {
        return composerSelection.has(nodeKey);
    }

    function toggleNode(nodeKey: string) {
        if (!isInterventableNode(nodeKey)) return; // Can't toggle non-interventable nodes
        const newSelection = new SvelteSet(composerSelection);
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
        const size = layer === "embed" || layer === "output" ? "small" : "large";
        tooltipPos = calcTooltipPos(event.clientX, event.clientY, size);
    }

    function handleNodeMouseLeave() {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        hoverTimeout = setTimeout(() => {
            if (!isHoveringTooltip) hoveredNode = null;
            hoverTimeout = null;
        }, 100);
    }

    function handleNodeClick(nodeKey: string) {
        toggleNode(nodeKey);
        hoveredNode = null;
    }

    // Drag-to-select handlers
    // Converts mouse event to SVG logical coordinates (accounting for zoom transform)
    function getSvgPoint(event: MouseEvent): { x: number; y: number } | null {
        if (!svgElement) return null;
        const container = svgElement.parentElement!;
        const rect = container.getBoundingClientRect();
        // Get container-relative position (with scroll offset)
        const containerX = event.clientX - rect.left + container.scrollLeft;
        const containerY = event.clientY - rect.top + container.scrollTop;
        // Convert to logical SVG coordinates by reversing the zoom transform
        return {
            x: (containerX - zoom.translateX) / zoom.scale,
            y: (containerY - zoom.translateY) / zoom.scale,
        };
    }

    function handlePanStart(event: MouseEvent) {
        // Pan on shift+left-click or middle-click
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
            return;
        }
        // Otherwise handle drag-to-select
        if (event.button === 0 && !event.shiftKey) {
            handleSvgMouseDown(event);
        }
    }

    function handleSvgMouseDown(event: MouseEvent) {
        const target = event.target as Element;
        if (target.closest(".node-group")) return;

        event.preventDefault();
        const point = getSvgPoint(event);
        if (!point) return;

        hoveredNode = null;
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
                const newSelection = new SvelteSet(composerSelection);
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

    function formatLogit(logit: number): string {
        return logit.toFixed(2);
    }

    function getRowLabel(layer: string): string {
        return _getRowLabel(layer);
    }
</script>

<div class="interventions-view">
    <!-- Composer Graph (Left) -->
    <div class="composer-graph">
        <!-- Shared view controls -->
        <ViewControls
            {topK}
            {componentGap}
            {layerGap}
            {filteredEdgeCount}
            {normalizeEdges}
            {ciThreshold}
            {hideUnpinnedEdges}
            {hideNodeCard}
            {onTopKChange}
            {onComponentGapChange}
            {onLayerGapChange}
            {onNormalizeChange}
            {onCiThresholdChange}
            {onHideUnpinnedEdgesChange}
            {onHideNodeCardChange}
        />

        <!-- Intervention controls -->
        <div class="intervention-controls">
            <span class="node-count">{selectedCount} / {interventableCount} selected</span>
            <span
                class="info-icon"
                data-tooltip="NOTE: Biases in each layer that have them are always active, regardless of which components are selected"
                >?</span
            >
            <div class="button-group">
                <button onclick={selectAll}>Select All</button>
                <button onclick={clearSelection}>Clear</button>
                <button
                    class="generate-btn"
                    onclick={onGenerateGraphFromSelection}
                    disabled={generatingSubgraph ||
                        selectedCount === 0 ||
                        (interventableCount > 0 && selectedCount === interventableCount)}
                    title={selectedCount === 0
                        ? "Select components to include in subgraph"
                        : "Generate a subgraph showing only attributions between selected components"}
                >
                    {generatingSubgraph ? "Generating..." : "Generate subgraph"}
                </button>
                <button class="run-btn" onclick={onRunIntervention} disabled={runningIntervention}>
                    {runningIntervention ? "Running..." : "Run forward pass"}
                </button>
            </div>
        </div>

        <!-- Graph wrapper for sticky layout -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
            class="graph-wrapper"
            class:panning={zoom.isPanning}
            onmousedown={handlePanStart}
            onmousemove={zoom.updatePan}
            onmouseup={zoom.endPan}
            onmouseleave={zoom.endPan}
        >
            <ZoomControls
                scale={zoom.scale}
                onZoomIn={zoom.zoomIn}
                onZoomOut={zoom.zoomOut}
                onReset={zoom.reset}
                hint="Shift+drag to pan, Shift+scroll to zoom"
            />

            <!-- Sticky layer labels (left) -->
            <div class="layer-labels-container" style="width: {LABEL_WIDTH}px;">
                <svg
                    width={LABEL_WIDTH}
                    height={layout.height * zoom.scale + Math.max(zoom.translateY, 0)}
                    style="display: block;"
                >
                    <g transform="translate(0, {zoom.translateY}) scale(1, {zoom.scale})">
                        {#each Object.entries(layout.layerYPositions) as [layer, y] (layer)}
                            {@const yCenter = y + COMPONENT_SIZE / 2}
                            <text
                                x={LABEL_WIDTH - 10}
                                y={yCenter}
                                text-anchor="end"
                                dominant-baseline="middle"
                                font-size="10"
                                font-weight="500"
                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                fill={colors.textSecondary}>{getRowLabel(layer)}</text
                            >
                        {/each}
                    </g>
                </svg>
            </div>

            <!-- Scrollable graph area -->
            <div class="graph-container" bind:this={graphContainer}>
                <svg
                    bind:this={svgElement}
                    width={layout.width * zoom.scale + Math.max(zoom.translateX, 0)}
                    height={layout.height * zoom.scale + Math.max(zoom.translateY, 0)}
                    style="display: block;"
                    onmousemove={handleSvgMouseMove}
                    onmouseup={handleSvgMouseUp}
                    onmouseleave={handleSvgMouseUp}
                >
                    <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
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

                        <!-- Cluster bars (below nodes) -->
                        <g class="cluster-bars-layer">
                            {#each layout.clusterSpans as span (`${span.layer}:${span.seqIdx}:${span.clusterId}`)}
                                {@const isHighlighted = hoveredClusterId === span.clusterId}
                                <!-- svelte-ignore a11y_no_static_element_interactions -->
                                <rect
                                    class="cluster-bar"
                                    class:highlighted={isHighlighted}
                                    x={span.xStart}
                                    y={span.y + CLUSTER_BAR_GAP}
                                    width={span.xEnd - span.xStart}
                                    height={CLUSTER_BAR_HEIGHT}
                                    rx="1"
                                    onmouseenter={() => (hoveredBarClusterId = span.clusterId)}
                                    onmouseleave={() => (hoveredBarClusterId = null)}
                                />
                            {/each}
                        </g>

                        <!-- Nodes -->
                        <g class="nodes-layer">
                            {#each allNodes as nodeKey (nodeKey)}
                                {@const pos = layout.nodePositions[nodeKey]}
                                {@const [layer, seqIdx, cIdx] = nodeKey.split(":")}
                                {@const interventable = isInterventableNode(nodeKey)}
                                {@const selected = interventable && isNodeSelected(nodeKey)}
                                {@const inSameCluster = isNodeInSameCluster(nodeKey)}
                                {@const isHoveredComponent = nodeMatchesHoveredComponent(nodeKey)}
                                {@const isDimmed =
                                    (hoveredNode !== null || hoveredBarClusterId !== null) &&
                                    !isHoveredComponent &&
                                    !inSameCluster &&
                                    !selected}
                                {#if pos}
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
                                            class:cluster-hovered={inSameCluster}
                                            class:dimmed={isDimmed}
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
                                fill={rgbaToCss(colors.positiveRgb, 0.1)}
                                stroke={colors.accent}
                                stroke-width="1"
                                stroke-dasharray="4 2"
                            />
                        {/if}
                    </g>
                </svg>

                <!-- Sticky token labels (bottom) -->
                <div class="token-labels-container">
                    <svg
                        width={layout.width * zoom.scale + Math.max(zoom.translateX, 0)}
                        height="50"
                        style="display: block;"
                    >
                        <g transform="translate({zoom.translateX}, 0) scale({zoom.scale}, 1)">
                            {#each tokens as token, i (i)}
                                {@const colCenter = layout.seqXStarts[i] + layout.seqWidths[i] / 2}
                                <text
                                    x={colCenter}
                                    y="20"
                                    text-anchor="middle"
                                    font-size="11"
                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                    font-weight="500"
                                    fill={colors.textPrimary}
                                    style="white-space: pre"
                                >
                                    {token}
                                </text>
                                <text
                                    x={colCenter}
                                    y="36"
                                    text-anchor="middle"
                                    font-size="9"
                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                    fill={colors.textMuted}>[{i}]</text
                                >
                            {/each}
                        </g>
                    </svg>
                </div>
            </div>
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
                    {@const isActive = activeRunId === run.id}
                    <div
                        class="run-card"
                        class:active={isActive}
                        class:flash={flashRunId === run.id}
                        role="button"
                        tabindex="0"
                        onclick={() => onSelectRun(run.id)}
                        onkeydown={(e) => e.key === "Enter" && onSelectRun(run.id)}
                    >
                        <div class="run-header">
                            <span class="run-time">{formatTime(run.created_at)}</span>
                            <span class="run-nodes">{run.selected_nodes.length} components</span>
                            <button
                                class="fork-btn"
                                title="Fork with modified tokens"
                                onclick={(e) => {
                                    e.stopPropagation();
                                    openForkModal(run.id);
                                }}>⑂</button
                            >
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
                                        <th class="rank-header">Input</th>
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
                                            <td class="rank-label">rank {rank + 1}</td>
                                            {#each run.result.predictions_per_position as preds, idx (idx)}
                                                {@const pred = preds[rank]}
                                                <td
                                                    class:has-pred={!!pred}
                                                    style={pred
                                                        ? `background: ${getOutputHeaderColor(pred.spd_prob)}`
                                                        : ""}
                                                >
                                                    {#if pred}
                                                        <span class="pred-token">"{pred.token}"</span>
                                                        <span class="pred-prob spd"
                                                            >SPD: {formatProb(pred.spd_prob)} (logit: {formatLogit(
                                                                pred.logit,
                                                            )})</span
                                                        >
                                                        <span class="pred-prob targ"
                                                            >Targ: {formatProb(pred.target_prob)} (logit: {formatLogit(
                                                                pred.target_logit,
                                                            )})</span
                                                        >
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

                        <!-- Forked runs -->
                        {#if run.forked_runs && run.forked_runs.length > 0}
                            <div class="forked-runs">
                                <div class="forked-runs-header">
                                    <span class="fork-icon">⑂</span>
                                    <span>{run.forked_runs.length} fork{run.forked_runs.length > 1 ? "s" : ""}</span>
                                </div>
                                {#each run.forked_runs as fork (fork.id)}
                                    <div class="forked-run-card">
                                        <div class="fork-header">
                                            <span class="fork-time">{formatTime(fork.created_at)}</span>
                                            <span class="fork-changes"
                                                >{fork.token_replacements.length} change{fork.token_replacements
                                                    .length > 1
                                                    ? "s"
                                                    : ""}</span
                                            >
                                            <button
                                                class="delete-btn"
                                                onclick={(e) => {
                                                    e.stopPropagation();
                                                    onDeleteFork(fork.id);
                                                }}>✕</button
                                            >
                                        </div>
                                        <!-- Mini logits for fork -->
                                        <div class="logits-mini">
                                            <table>
                                                <thead>
                                                    <tr>
                                                        <th class="rank-header">Input</th>
                                                        {#each fork.result.input_tokens as token, idx (idx)}
                                                            {@const isChanged = fork.token_replacements.some(
                                                                (r) => r[0] === idx,
                                                            )}
                                                            <th title={token} class:changed={isChanged}>
                                                                <span class="token-text">"{token}"</span>
                                                            </th>
                                                        {/each}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {#each Array(Math.min(3, MAX_PREDICTIONS)) as _, rank (rank)}
                                                        <tr>
                                                            <td class="rank-label">rank {rank + 1}</td>
                                                            {#each fork.result.predictions_per_position as preds, idx (idx)}
                                                                {@const pred = preds[rank]}
                                                                <td
                                                                    class:has-pred={!!pred}
                                                                    style={pred
                                                                        ? `background: ${getOutputHeaderColor(pred.spd_prob)}`
                                                                        : ""}
                                                                >
                                                                    {#if pred}
                                                                        <span class="pred-token">"{pred.token}"</span>
                                                                        <span class="pred-prob spd"
                                                                            >SPD: {formatProb(pred.spd_prob)} (logit: {formatLogit(
                                                                                pred.logit,
                                                                            )})</span
                                                                        >
                                                                        <span class="pred-prob targ"
                                                                            >Targ: {formatProb(pred.target_prob)} (logit:
                                                                            {formatLogit(pred.target_logit)})</span
                                                                        >
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
                {/each}
            </div>
        {/if}
    </div>

    <!-- Node tooltip -->
    {#if hoveredNode}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            {hideNodeCard}
            outputProbs={graph.data.outputProbs}
            nodeCiVals={graph.data.nodeCiVals}
            nodeSubcompActs={graph.data.nodeSubcompActs}
            {tokens}
            edgesBySource={graph.data.edgesBySource}
            edgesByTarget={graph.data.edgesByTarget}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
        />
    {/if}

    <!-- Fork Modal -->
    {#if forkingRunId !== null}
        <div
            class="modal-overlay"
            onclick={closeForkModal}
            onkeydown={(e) => e.key === "Escape" && closeForkModal()}
            role="dialog"
            aria-modal="true"
            tabindex="-1"
        >
            <div class="fork-modal" onclick={(e) => e.stopPropagation()} onkeydown={(e) => e.stopPropagation()}>
                <div class="modal-header">
                    <h3>Fork Run</h3>
                    <button class="close-btn" onclick={closeForkModal}>✕</button>
                </div>
                <div class="modal-body">
                    <p class="modal-description">
                        Replace tokens and run the same subnetwork. Changes: {forkReplacements.length}
                    </p>
                    <div class="token-editor">
                        {#each forkSlotStates as slot, idx (idx)}
                            {@const isReplaced = slot.tokenId !== null && slot.tokenId !== tokenIds[idx]}
                            <div class="token-slot" class:replaced={isReplaced}>
                                <span class="token-label">pos {idx}: "{tokens[idx]}"</span>
                                <TokenDropdown
                                    value={slot.value}
                                    selectedTokenId={slot.tokenId}
                                    onSelect={(tokenId, tokenString) => handleForkSlotSelect(idx, tokenId, tokenString)}
                                    placeholder="Search..."
                                />
                                {#if isReplaced}
                                    <button class="reset-btn" onclick={() => resetForkSlot(idx)}>↩</button>
                                {/if}
                            </div>
                        {/each}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="cancel-btn" onclick={closeForkModal}>Cancel</button>
                    <button
                        class="submit-btn"
                        disabled={forkReplacements.length === 0 || forkingInProgress}
                        onclick={submitFork}
                    >
                        {forkingInProgress ? "Running..." : "Fork & Run"}
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .interventions-view {
        display: flex;
        flex: 1;
        min-height: 0;
        gap: var(--space-4);
    }

    /* Composer Graph */
    .composer-graph {
        display: flex;
        flex-direction: column;
        border: 1px solid var(--border-default);
        background: var(--bg-surface);
        overflow: hidden;
    }

    .intervention-controls {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
    }

    .node-count {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .button-group {
        display: flex;
        gap: var(--space-2);
        margin-left: auto;
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

    .generate-btn {
        background: var(--status-info) !important;
        color: white !important;
        border-color: var(--status-info) !important;
    }

    .generate-btn:hover:not(:disabled) {
        filter: brightness(1.1);
    }

    .generate-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
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

    .graph-wrapper {
        display: flex;
        overflow: hidden;
        position: relative;
    }

    .graph-wrapper.panning {
        cursor: grabbing;
    }

    .layer-labels-container {
        position: sticky;
        left: 0;
        background: var(--bg-surface);
        border-right: 1px solid var(--border-default);
        z-index: 11;
        flex-shrink: 0;
    }

    .graph-container {
        overflow: auto;
        flex: 1;
        position: relative;
        background: var(--bg-inset);
    }

    .token-labels-container {
        position: sticky;
        bottom: 0;
        background: var(--bg-surface);
        border-top: 1px solid var(--border-default);
        z-index: 10;
    }

    .graph-container svg {
        cursor: crosshair;
    }

    .node-group {
        cursor: pointer;
    }

    .node-group .node {
        transform-box: fill-box;
        transform-origin: center;
        transition:
            opacity var(--transition-fast),
            fill var(--transition-fast),
            transform var(--transition-normal);
    }

    .node-group .node.cluster-hovered {
        transform: rotate(45deg);
    }

    .node-group .node.dimmed {
        transform: scale(0.5);
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

    .cluster-bar {
        fill: var(--text-secondary);
        opacity: 0.5;
        cursor: pointer;
        transition:
            opacity var(--transition-normal),
            fill var(--transition-normal);
    }

    .cluster-bar:hover,
    .cluster-bar.highlighted {
        fill: var(--text-primary);
        opacity: 0.8;
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
        transition: border-color var(--transition-fast);
    }

    .run-card:hover {
        border-color: var(--border-strong);
    }

    .run-card.active {
        border-color: var(--accent-primary);
        background: var(--bg-inset);
    }

    .run-card.flash {
        animation: flash-new 1.5s ease-out;
    }

    @keyframes flash-new {
        0% {
            background: var(--accent-primary);
        }
        100% {
            background: var(--bg-elevated);
        }
    }

    .run-card.active.flash {
        animation: flash-new-active 1.5s ease-out;
    }

    @keyframes flash-new-active {
        0% {
            background: var(--accent-primary);
        }
        100% {
            background: var(--bg-inset);
        }
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

    .fork-btn,
    .delete-btn {
        padding: 2px 6px;
        background: transparent;
        border: none;
        color: var(--text-muted);
        font-size: var(--text-xs);
        cursor: pointer;
    }

    .fork-btn:hover {
        color: var(--text-primary);
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

    .logits-mini th.rank-header {
        font-weight: 600;
        color: var(--text-primary);
    }

    .logits-mini .token-text {
        font-size: 9px;
    }

    .logits-mini td.rank-label {
        background: var(--bg-surface);
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 9px;
        text-align: left;
        padding-left: 6px;
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

    .pred-prob.spd {
        color: var(--text-secondary);
    }

    .pred-prob.targ {
        color: var(--text-secondary);
    }

    /* Forked runs */
    .forked-runs {
        margin-top: var(--space-2);
        padding-top: var(--space-2);
        border-top: 1px dashed var(--border-subtle);
    }

    .forked-runs-header {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-xs);
        color: var(--text-muted);
        margin-bottom: var(--space-1);
    }

    .fork-icon {
        font-size: var(--text-sm);
    }

    .forked-run-card {
        background: var(--bg-inset);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: var(--space-1);
        margin-bottom: var(--space-1);
    }

    .fork-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-xs);
        margin-bottom: var(--space-1);
    }

    .fork-time {
        color: var(--text-muted);
    }

    .fork-changes {
        color: var(--status-info);
        margin-left: auto;
    }

    .logits-mini th.changed {
        background: rgba(var(--status-info-rgb, 59, 130, 246), 0.2);
    }

    /* Fork Modal */
    .modal-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }

    .fork-modal {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        min-width: 400px;
        max-width: 90vw;
        max-height: 90vh;
        display: flex;
        flex-direction: column;
    }

    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-3);
        border-bottom: 1px solid var(--border-subtle);
    }

    .modal-header h3 {
        margin: 0;
        font-size: var(--text-base);
        font-weight: 600;
    }

    .close-btn {
        background: transparent;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: var(--text-base);
    }

    .close-btn:hover {
        color: var(--text-primary);
    }

    .modal-body {
        padding: var(--space-3);
        overflow-y: auto;
    }

    .modal-description {
        margin: 0 0 var(--space-3) 0;
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }

    .token-editor {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
    }

    .token-slot {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding: var(--space-1);
        background: var(--bg-inset);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
    }

    .token-slot.replaced {
        border-color: var(--status-info);
        background: rgba(var(--status-info-rgb), 0.1);
    }

    .token-label {
        font-size: 9px;
        color: var(--text-muted);
        font-family: var(--font-mono);
        white-space: pre;
    }

    .reset-btn {
        background: transparent;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 10px;
        padding: 0;
    }

    .reset-btn:hover {
        color: var(--text-primary);
    }

    .modal-footer {
        display: flex;
        justify-content: flex-end;
        gap: var(--space-2);
        padding: var(--space-3);
        border-top: 1px solid var(--border-subtle);
    }

    .cancel-btn,
    .submit-btn {
        padding: var(--space-1) var(--space-3);
        border-radius: var(--radius-sm);
        font-size: var(--text-sm);
        cursor: pointer;
    }

    .cancel-btn {
        background: transparent;
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .cancel-btn:hover {
        background: var(--bg-hover);
    }

    .submit-btn {
        background: var(--status-info);
        border: none;
        color: white;
    }

    .submit-btn:hover:not(:disabled) {
        filter: brightness(1.1);
    }

    .submit-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
</style>
