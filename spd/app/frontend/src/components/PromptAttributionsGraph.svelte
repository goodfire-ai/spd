<script lang="ts">
    import { getContext } from "svelte";
    import { SvelteSet } from "svelte/reactivity";
    import type { GraphData, PinnedNode, HoveredNode, HoveredEdge, NodePosition } from "../lib/promptAttributionsTypes";
    import { colors, getEdgeColor, getSubcompActColor, rgbToCss, getNextTokenProbBgColor } from "../lib/colors";
    import { displaySettings } from "../lib/displaySettings.svelte";
    import {
        lerp,
        calcTooltipPos,
        sortComponentsByImportance,
        sortComponentsByCluster,
        computeComponentOffsets,
        computeClusterSpans,
        type ClusterSpan,
        type TooltipPos,
    } from "./prompt-attr/graphUtils";
    import NodeTooltip from "./prompt-attr/NodeTooltip.svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import { useZoomPan } from "../lib/useZoomPan.svelte";
    import ZoomControls from "../lib/ZoomControls.svelte";
    import {
        parseLayer,
        getRowKey as _getRowKey,
        getRowLabel as _getRowLabel,
        sortRows,
        getGroupProjections,
        buildLayerAddress,
    } from "../lib/graphLayout";

    const runState = getContext<RunContext>(RUN_KEY);

    // Constants
    const COMPONENT_SIZE = 8;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;
    const CLUSTER_BAR_HEIGHT = 3;
    const CLUSTER_BAR_GAP = 2;
    const LAYER_X_OFFSET = 3; // Horizontal offset per layer to avoid edge overlap

    type Props = {
        data: GraphData;
        tokenIds: number[];
        topK: number;
        componentGap: number;
        layerGap: number;
        hideUnpinnedEdges: boolean;
        hideNodeCard: boolean;
        stagedNodes: PinnedNode[];
        onStagedNodesChange: (nodes: PinnedNode[]) => void;
        onEdgeCountChange?: (count: number) => void;
    };

    let {
        data,
        tokenIds,
        topK,
        componentGap,
        layerGap,
        hideUnpinnedEdges,
        hideNodeCard,
        stagedNodes,
        onStagedNodesChange,
        onEdgeCountChange,
    }: Props = $props();

    // Compute masked prediction probability of self given previous position.
    // For token at position i, we look up outputProbs[(i-1):tokenIds[i]] - how well
    // position i-1 predicted this token. First token has no previous, so null.
    // NOTE: outputProbs only includes tokens with >=1% probability (backend threshold).
    // If the correct token isn't found, it means the masked model gave it <1% probability.
    const maskedSelfProbs = $derived.by(() => {
        const probs: (number | null)[] = [];
        for (let i = 0; i < data.tokens.length; i++) {
            if (i === 0) {
                probs.push(null); // First token has no previous position
            } else {
                const thisTokenId = tokenIds[i];
                const entry = data.outputProbs[`${i - 1}:${thisTokenId}`];
                probs.push(entry?.prob ?? null);
            }
        }
        return probs;
    });

    // UI state
    let hoveredNode = $state<HoveredNode | null>(null);
    let hoveredEdge = $state<HoveredEdge | null>(null);
    let hoveredBarClusterId = $state<number | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state<TooltipPos>({ left: 0, top: 0 });
    let edgeTooltipPos = $state({ x: 0, y: 0 });

    // Alt/Option key temporarily toggles hide unpinned edges
    let altHeld = $state(false);
    const effectiveHideUnpinned = $derived(altHeld ? !hideUnpinnedEdges : hideUnpinnedEdges);

    $effect(() => {
        function onKeyDown(e: KeyboardEvent) {
            if (e.key === "Alt") {
                altHeld = true;
            }
        }
        function onKeyUp(e: KeyboardEvent) {
            if (e.key === "Alt") {
                altHeld = false;
            }
        }
        function onBlur() {
            altHeld = false;
        }
        window.addEventListener("keydown", onKeyDown);
        window.addEventListener("keyup", onKeyUp);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKeyDown);
            window.removeEventListener("keyup", onKeyUp);
            window.removeEventListener("blur", onBlur);
        };
    });

    // Refs
    let graphContainer: HTMLDivElement;
    let innerContainer: HTMLDivElement;

    // Zoom/pan
    const zoom = useZoomPan(() => innerContainer);

    function getRowKey(layer: string): string {
        return _getRowKey(layer);
    }

    function getRowLabel(layer: string): string {
        return _getRowLabel(layer);
    }

    // Use pre-computed values from backend, derive max CI
    const maxAbsAttr = $derived(data.maxAbsAttr || 1);
    const maxCi = $derived.by(() => {
        let max = 0;
        for (const ci of Object.values(data.nodeCiVals)) {
            if (ci > max) max = ci;
        }
        return max || 1; // Avoid division by zero
    });
    // Check if nodeSubcompActs has actual data (empty object {} is truthy in JS)
    const hasSubcompActData = $derived(data.nodeSubcompActs && Object.keys(data.nodeSubcompActs).length > 0);
    const maxAbsSubcompAct = $derived(data.maxAbsSubcompAct);

    // All nodes from nodeCiVals (for layout and rendering)
    const allNodes = $derived(new SvelteSet(Object.keys(data.nodeCiVals)));

    // Pre-compute pinned node keys for efficient lookup
    const pinnedNodeKeys = $derived(new Set(stagedNodes.map((p) => `${p.layer}:${p.seqIdx}:${p.cIdx}`)));

    // For hover, we match by component (layer:cIdx), ignoring seqIdx
    const hoveredComponentKey = $derived(hoveredNode ? `${hoveredNode.layer}:${hoveredNode.cIdx}` : null);

    // Get cluster ID of hovered node or bar (for cluster-wide rotation effect)
    const hoveredClusterId = $derived.by(() => {
        if (hoveredBarClusterId !== null) return hoveredBarClusterId;
        if (!hoveredNode) return undefined;
        return runState.getClusterId(hoveredNode.layer, hoveredNode.cIdx);
    });

    // Filter edges by topK (for rendering)
    const filteredEdges = $derived.by(() => {
        const edgesCopy = [...data.edges];
        const sortedEdges = edgesCopy.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
        return sortedEdges.slice(0, topK);
    });

    // Build layout
    const { nodePositions, layerYPositions, seqXStarts, width, height, clusterSpans } = $derived.by(() => {
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

        // Sort rows for Y positioning
        const rows = sortRows(Array.from(allRows));

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
            if (distanceFromOutput === 0 || layer === "wte") {
                layerXOffsets[layer] = 0;
            } else {
                layerXOffsets[layer] = distanceFromOutput % 2 === 1 ? LAYER_X_OFFSET : -LAYER_X_OFFSET;
            }
        }

        // Calculate column widths
        const tokens = data.tokens;
        const maxComponentsPerSeq = tokens.map((_, seqIdx) => {
            let maxAtSeq = 0;
            for (const row of rows) {
                // Count nodes in this row at this seq position
                // Rows are "block.sublayer" — find all layers that belong to this row
                let totalInRow = 0;
                for (const layer of allLayers) {
                    if (getRowKey(layer) === row) {
                        const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                        totalInRow += nodes.length;
                    }
                }
                // Add gaps between grouped projections (only for grouped rows)
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

        const MIN_COL_WIDTH = 30;
        const COL_PADDING = 16;
        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + componentGap) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Position nodes and compute cluster spans
        const nodePositions: Record<string, NodePosition> = {};
        const allClusterSpans: ClusterSpan[] = [];
        const QKV_GROUP_GAP = COMPONENT_SIZE + componentGap;

        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const groupProjs = info.sublayer ? getGroupProjections(info.sublayer) : null;
            const isGrouped = groupProjs !== null && info.projection !== null && groupProjs.includes(info.projection);

            for (let seqIdx = 0; seqIdx < tokens.length; seqIdx++) {
                const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!nodes) continue;

                let baseX = seqXStarts[seqIdx] + COL_PADDING + layerXOffsets[layer];
                const baseY = layerYPositions[layer];

                // For grouped projections (e.g. q/k/v), offset X based on position in group
                if (isGrouped && groupProjs && info.projection) {
                    const projIdx = groupProjs.indexOf(info.projection);
                    for (let i = 0; i < projIdx; i++) {
                        const prevLayer = buildLayerAddress(info.block, info.sublayer, groupProjs[i]);
                        const prevLayerNodes = nodesPerLayerSeq[`${prevLayer}:${seqIdx}`] ?? [];
                        baseX += prevLayerNodes.length * (COMPONENT_SIZE + componentGap);
                        baseX += QKV_GROUP_GAP;
                    }
                }

                // Output nodes always sort by probability; internal nodes sort by cluster if mapping loaded, else by CI
                const sorted =
                    layer === "output" || !runState.clusterMapping
                        ? sortComponentsByImportance(nodes, layer, seqIdx, data.nodeCiVals, data.outputProbs)
                        : sortComponentsByCluster(nodes, layer, seqIdx, data.nodeCiVals, runState.getClusterId);
                const offsets = computeComponentOffsets(sorted, COMPONENT_SIZE, componentGap);

                for (const cIdx of nodes) {
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
        const widthVal = totalSeqWidth + MARGIN.right;
        const maxY = Math.max(...Object.values(layerYPositions), 0) + COMPONENT_SIZE;
        const heightVal = maxY + MARGIN.bottom;

        return {
            nodePositions,
            layerYPositions,
            seqXStarts,
            width: widthVal,
            height: heightVal,
            clusterSpans: allClusterSpans,
        };
    });

    // Derived SVG dimensions (fixes negative translate bug)
    const svgWidth = $derived(width * zoom.scale + Math.max(zoom.translateX, 0));
    const svgHeight = $derived(height * zoom.scale + Math.max(zoom.translateY, 0));

    const EDGE_HIT_AREA_WIDTH = 4; // Wider invisible stroke for easier hover

    // Check if a node key matches the currently hovered component (same layer:cIdx, any seqIdx)
    // For wte nodes: match by token value (highlight same tokens across positions)
    // For other nodes: match by layer:cIdx (highlight same component across positions)
    function nodeMatchesHoveredComponent(nodeKey: string): boolean {
        if (!hoveredNode) return false;
        const [layer, seqIdxStr, cIdx] = nodeKey.split(":");
        const seqIdx = parseInt(seqIdxStr);

        // For wte nodes, match by token value
        if (hoveredNode.layer === "wte") {
            if (layer !== "wte") return false;
            return data.tokens[seqIdx] === data.tokens[hoveredNode.seqIdx];
        }

        // For other nodes, match by component key (layer:cIdx)
        return `${layer}:${cIdx}` === hoveredComponentKey;
    }

    // Check if a node is in the same cluster as the hovered node (for cluster rotation effect)
    function isNodeInSameCluster(nodeKey: string): boolean {
        // Only trigger if hovered node has a numeric cluster ID (not singleton/no mapping)
        if (hoveredClusterId === undefined || hoveredClusterId === null) return false;
        const [layer, , cIdxStr] = nodeKey.split(":");
        const cIdx = parseInt(cIdxStr);
        const nodeClusterId = runState.getClusterId(layer, cIdx);
        return nodeClusterId === hoveredClusterId;
    }

    type EdgeState = "normal" | "highlighted" | "hidden";

    // Hover acts as a "promotion": hidden → normal → highlighted
    function getEdgeState(src: string, tgt: string): EdgeState {
        const hasPinned = pinnedNodeKeys.size > 0;
        const connectedToPinned = pinnedNodeKeys.has(src) || pinnedNodeKeys.has(tgt);
        const connectedToHoveredNode = nodeMatchesHoveredComponent(src) || nodeMatchesHoveredComponent(tgt);
        const isThisEdgeHovered = hoveredEdge?.src === src && hoveredEdge?.tgt === tgt;

        // No pinned nodes - just node hover behavior
        if (!hasPinned) {
            return connectedToHoveredNode ? "highlighted" : "normal";
        }

        // Has pinned nodes
        if (effectiveHideUnpinned) {
            if (!connectedToPinned) {
                // Show (not highlighted) edges connected to hovered component
                if (connectedToHoveredNode) return "normal";
                return "hidden";
            }
            // Highlight edges connected to pinned on edge/node hover
            if (isThisEdgeHovered || connectedToHoveredNode) return "highlighted";
            return "normal";
        } else {
            // Show all edges, connected ones highlighted by default
            // Edge hover: only that edge highlighted, others normal
            if (hoveredEdge) {
                return isThisEdgeHovered ? "highlighted" : "normal";
            }
            // Node hover: highlight connected to hovered component OR pinned nodes
            if (hoveredNode) {
                return connectedToHoveredNode || connectedToPinned ? "highlighted" : "normal";
            }
            // No hover: connected to pinned are highlighted
            return connectedToPinned ? "highlighted" : "normal";
        }
    }

    // Build SVG edges string (for {@html} - performance optimization)
    // Render order: visible paths first (smaller on top), then hit areas (larger on top)
    // Only render hit areas for edges connected to pinned nodes
    const edgesSvgString = $derived.by(() => {
        let visibleSvg = "";
        let hitAreaSvg = "";

        // filteredEdges is already sorted by abs(val) descending
        // Render visible paths in reverse order (smallest first, so largest renders on top)
        for (let i = filteredEdges.length - 1; i >= 0; i--) {
            const edge = filteredEdges[i];
            const p1 = nodePositions[edge.src];
            const p2 = nodePositions[edge.tgt];
            if (p1 && p2) {
                const color = getEdgeColor(edge.val);
                const w = lerp(1, 4, Math.abs(edge.val) / maxAbsAttr);
                const op = lerp(0, 0.5, Math.abs(edge.val) / maxAbsAttr);
                const dy = Math.abs(p2.y - p1.y);
                const curveOffset = Math.max(20, dy * 0.4);
                const cp1y = p1.y - curveOffset;
                const cp2y = p2.y + curveOffset;
                const d = `M ${p1.x},${p1.y} C ${p1.x},${cp1y} ${p2.x},${cp2y} ${p2.x},${p2.y}`;
                visibleSvg += `<path class="edge edge-visible" data-src="${edge.src}" data-tgt="${edge.tgt}" data-val="${edge.val}" d="${d}" stroke="${color}" stroke-width="${w}" opacity="${op}" fill="none" pointer-events="none"/>`;
            }
        }

        // Only render hit areas for edges connected to pinned nodes
        // Render in reverse order so largest edges' hit areas are on top
        for (let i = filteredEdges.length - 1; i >= 0; i--) {
            const edge = filteredEdges[i];
            if (!pinnedNodeKeys.has(edge.src) && !pinnedNodeKeys.has(edge.tgt)) continue;

            const p1 = nodePositions[edge.src];
            const p2 = nodePositions[edge.tgt];
            if (p1 && p2) {
                const dy = Math.abs(p2.y - p1.y);
                const curveOffset = Math.max(20, dy * 0.4);
                const cp1y = p1.y - curveOffset;
                const cp2y = p2.y + curveOffset;
                const d = `M ${p1.x},${p1.y} C ${p1.x},${cp1y} ${p2.x},${cp2y} ${p2.x},${p2.y}`;
                hitAreaSvg += `<path class="edge edge-hit-area" data-src="${edge.src}" data-tgt="${edge.tgt}" data-val="${edge.val}" d="${d}" stroke="transparent" stroke-width="${EDGE_HIT_AREA_WIDTH}" fill="none"/>`;
            }
        }

        return visibleSvg + hitAreaSvg;
    });

    // Check if a node key should be highlighted (pinned or hovered component)
    function isNodeHighlighted(nodeKey: string): boolean {
        return pinnedNodeKeys.has(nodeKey) || nodeMatchesHoveredComponent(nodeKey);
    }

    // Pre-compute node styles (fill, opacity) - only recomputes when data/layout changes, not on hover
    const nodeStyles = $derived.by(() => {
        const styles: Record<string, { fill: string; opacity: number }> = {};

        for (const nodeKey of Object.keys(nodePositions)) {
            const [layer, seqIdxStr, cIdxStr] = nodeKey.split(":");
            const seqIdx = parseInt(seqIdxStr);
            const cIdx = parseInt(cIdxStr);

            let fill: string = colors.nodeDefault;
            let opacity = 0.2;

            if (layer === "output") {
                const probEntry = data.outputProbs[`${seqIdx}:${cIdx}`];
                if (probEntry) {
                    fill = rgbToCss(colors.outputBase);
                    opacity = 0.2 + probEntry.prob * 0.8;
                } else {
                    // remove me. we should just assert this should be present
                    console.error(`OutputNodeCard: no entry for ${seqIdx}:${cIdx}`);
                }
            } else {
                // Component nodes: color/opacity based on CI or subcomp activation
                if (displaySettings.nodeColorMode === "ci" || !hasSubcompActData) {
                    const ci = data.nodeCiVals[`${layer}:${seqIdx}:${cIdx}`] || 0;
                    const intensity = ci / maxCi;
                    if (intensity > 1) {
                        throw new Error(`Inconsistent state: intensity > 1: ${intensity}`);
                    }
                    opacity = 0.2 + intensity * 0.8;
                } else {
                    const subcompAct = data.nodeSubcompActs![`${layer}:${seqIdx}:${cIdx}`] ?? 0;
                    const intensity = subcompAct / maxAbsSubcompAct;
                    if (intensity > 1) {
                        throw new Error(`Inconsistent state: intensity > 1: ${intensity}`);
                    }
                    fill = getSubcompActColor(subcompAct);
                    opacity = 0.2 + intensity * 0.8;
                }
            }

            styles[nodeKey] = { fill, opacity };
        }

        return styles;
    });

    // Event handlers
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    function handleNodeMouseEnter(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        // Clear any pending leave timeout
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }

        hoveredNode = { layer, seqIdx, cIdx };
        const size = layer === "wte" || layer === "output" ? "small" : "large";
        tooltipPos = calcTooltipPos(event.clientX, event.clientY, size);
    }

    function handleNodeMouseLeave() {
        // Clear any existing timeout first
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
        }
        hoverTimeout = setTimeout(() => {
            if (!isHoveringTooltip) {
                hoveredNode = null;
            }
            hoverTimeout = null;
        }, 100);
    }

    function handleNodeClick(layer: string, seqIdx: number, cIdx: number) {
        toggleComponentPinned(layer, cIdx, seqIdx);
        hoveredNode = null;
    }

    function toggleComponentPinned(layer: string, cIdx: number, seqIdx: number) {
        const idx = stagedNodes.findIndex((p) => p.layer === layer && p.seqIdx === seqIdx && p.cIdx === cIdx);
        if (idx >= 0) {
            onStagedNodesChange(stagedNodes.filter((_, i) => i !== idx));
        } else {
            onStagedNodesChange([...stagedNodes, { layer, seqIdx, cIdx }]);
        }
    }

    function handleEdgeMouseEnter(event: MouseEvent) {
        const target = event.target as SVGElement;
        if (target.classList.contains("edge-hit-area")) {
            const src = target.getAttribute("data-src") || "";
            const tgt = target.getAttribute("data-tgt") || "";
            const val = parseFloat(target.getAttribute("data-val") || "0");
            hoveredEdge = { src, tgt, val };
            edgeTooltipPos = { x: event.clientX + 10, y: event.clientY + 10 };
        }
    }

    function handleEdgeMouseLeave(event: MouseEvent) {
        const target = event.target as SVGElement;
        if (target.classList.contains("edge-hit-area")) {
            hoveredEdge = null;
        }
    }

    function handlePanStart(event: MouseEvent) {
        const target = event.target as Element;
        if (target.closest(".node-group") || target.closest(".cluster-bar")) return;
        // Pan on shift+left-click or middle-click
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
        }
    }

    // Update edge classes based on state (DOM manipulation for performance with @html edges)
    $effect(() => {
        if (!graphContainer) return;

        const edges = graphContainer.querySelectorAll(".edge-visible");
        edges.forEach((el) => {
            const src = el.getAttribute("data-src") || "";
            const tgt = el.getAttribute("data-tgt") || "";
            const state = getEdgeState(src, tgt);

            el.classList.toggle("highlighted", state === "highlighted");
            el.classList.toggle("hidden", state === "hidden");
        });
    });

    // Notify parent of edge count changes
    $effect(() => {
        onEdgeCountChange?.(filteredEdges.length);
    });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
    class="graph-wrapper"
    class:panning={zoom.isPanning}
    bind:this={graphContainer}
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

    <div class="layer-labels-container" style="width: {LABEL_WIDTH}px;">
        <svg width={LABEL_WIDTH} height={svgHeight} style="display: block;">
            <g transform="translate(0, {zoom.translateY}) scale(1, {zoom.scale})">
                {#each Object.entries(layerYPositions) as [layer, y] (layer)}
                    {@const yCenter = y + COMPONENT_SIZE / 2}
                    <text
                        x={LABEL_WIDTH - 10}
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
            </g>
        </svg>
    </div>

    <div class="graph-container" bind:this={innerContainer}>
        <svg width={svgWidth} height={svgHeight} onmouseover={handleEdgeMouseEnter} onmouseout={handleEdgeMouseLeave}>
            <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
                <!-- Edges (bulk rendered for performance, uses @html for large SVG performance) -->
                <g class="edges-layer">
                    <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                    {@html edgesSvgString}
                </g>

                <!-- Cluster bars (below nodes) -->
                <g class="cluster-bars-layer">
                    {#each clusterSpans as span (`${span.layer}:${span.seqIdx}:${span.clusterId}`)}
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

                <!-- Nodes (reactive for interactivity) -->
                <g class="nodes-layer">
                    {#each Object.entries(nodePositions) as [key, pos] (key)}
                        {@const [layer, seqIdxStr, cIdxStr] = key.split(":")}
                        {@const seqIdx = parseInt(seqIdxStr)}
                        {@const cIdx = parseInt(cIdxStr)}
                        {@const isHighlighted = isNodeHighlighted(key)}
                        {@const isPinned = pinnedNodeKeys.has(key)}
                        {@const inSameCluster = isNodeInSameCluster(key)}
                        {@const isHoveredComponent = nodeMatchesHoveredComponent(key)}
                        {@const isDimmed =
                            (hoveredNode !== null || hoveredBarClusterId !== null) &&
                            !isHoveredComponent &&
                            !inSameCluster &&
                            !isPinned}
                        {@const style = nodeStyles[key]}
                        <g
                            class="node-group"
                            onmouseenter={(e) => handleNodeMouseEnter(e, layer, seqIdx, cIdx)}
                            onmouseleave={handleNodeMouseLeave}
                            onclick={() => handleNodeClick(layer, seqIdx, cIdx)}
                        >
                            <!-- Invisible hit area for easier hovering -->
                            <rect
                                x={pos.x - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                y={pos.y - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                width={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                height={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                fill="transparent"
                            />
                            <!-- Visible node -->
                            <rect
                                class="node"
                                class:highlighted={isHighlighted}
                                class:cluster-hovered={inSameCluster}
                                class:dimmed={isDimmed}
                                x={pos.x - COMPONENT_SIZE / 2}
                                y={pos.y - COMPONENT_SIZE / 2}
                                width={COMPONENT_SIZE}
                                height={COMPONENT_SIZE}
                                fill={style.fill}
                                rx="1"
                                opacity={style.opacity}
                            />
                        </g>
                    {/each}
                </g>
            </g>
        </svg>

        <div class="token-labels-container">
            <svg width={svgWidth} height="50" style="display: block;">
                <g transform="translate({zoom.translateX}, 0) scale({zoom.scale}, 1)">
                    {#each data.tokens as token, i (i)}
                        {@const colLeft = seqXStarts[i] + 8}
                        {@const maskedProb = maskedSelfProbs[i]}
                        <text
                            x={colLeft}
                            y="20"
                            text-anchor="start"
                            font-size="11"
                            font-family="'Berkeley Mono', 'SF Mono', monospace"
                            font-weight="500"
                            fill={colors.textPrimary}
                            style="white-space: pre"
                        >
                            {token}
                        </text>
                        <text
                            x={colLeft}
                            y="36"
                            text-anchor="start"
                            font-size="9"
                            font-family="'Berkeley Mono', 'SF Mono', monospace"
                            fill={colors.textMuted}>[{i}]</text
                        >
                        <!-- Masked prediction probability dot: P(self | previous) -->
                        {@const isFirstToken = i === 0}
                        <circle
                            cx={colLeft + 24}
                            cy="33"
                            r="4"
                            fill={getNextTokenProbBgColor(maskedProb)}
                            stroke={colors.textMuted}
                            stroke-width="0.5"
                        >
                            <title
                                >{maskedProb !== null
                                    ? `P(self): ${(maskedProb * 100).toFixed(1)}%`
                                    : isFirstToken
                                      ? "First token"
                                      : "P(self): <1%"}</title
                            >
                        </circle>
                    {/each}
                </g>
            </svg>
        </div>
    </div>

    <!-- Edge tooltip -->
    {#if hoveredEdge}
        <div class="edge-tooltip" style="left: {edgeTooltipPos.x}px; top: {edgeTooltipPos.y}px;">
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Src</span>
                <code>{hoveredEdge.src}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Tgt</span>
                <code>{hoveredEdge.tgt}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Val</span>
                <span style="color: {getEdgeColor(hoveredEdge.val)}; font-weight: 600;">
                    {hoveredEdge.val.toFixed(4)}
                </span>
            </div>
        </div>
    {/if}

    <!-- Node tooltip -->
    {#if hoveredNode}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            {hideNodeCard}
            outputProbs={data.outputProbs}
            nodeCiVals={data.nodeCiVals}
            nodeSubcompActs={data.nodeSubcompActs}
            tokens={data.tokens}
            edgesBySource={data.edgesBySource}
            edgesByTarget={data.edgesByTarget}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
            onPinComponent={toggleComponentPinned}
        />
    {/if}
</div>

<style>
    .graph-wrapper {
        display: flex;
        background: var(--bg-surface);
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

    svg {
        display: block;
    }

    :global(.edge.highlighted) {
        opacity: 1 !important;
        stroke-width: 3 !important;
    }

    :global(.edge.hidden) {
        display: none;
    }

    .node-group {
        cursor: pointer;
    }

    .node {
        transform-box: fill-box;
        transform-origin: center;
        transition: transform var(--transition-normal);
    }

    .node.cluster-hovered {
        transform: rotate(45deg);
    }

    .node.dimmed {
        transform: scale(0.5);
    }

    .node.highlighted {
        stroke: var(--accent-primary) !important;
        stroke-width: 2px !important;
        filter: brightness(1.2);
        opacity: 1 !important;
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

    .edge-tooltip {
        position: fixed;
        padding: var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        z-index: 1000;
        pointer-events: auto;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
    }

    .edge-tooltip-row {
        margin: var(--space-1) 0;
        display: flex;
        gap: var(--space-2);
    }

    .edge-tooltip-label {
        color: var(--text-muted);
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
        min-width: 4em;
    }

    .edge-tooltip code {
        color: var(--text-primary);
        font-size: var(--text-sm);
    }
</style>
