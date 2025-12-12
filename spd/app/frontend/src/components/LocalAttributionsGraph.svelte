<script lang="ts">
    import { SvelteSet } from "svelte/reactivity";
    import type {
        GraphData,
        ActivationContextsSummary,
        PinnedNode,
        HoveredNode,
        HoveredEdge,
        LayerInfo,
        NodePosition,
        ComponentDetail,
    } from "../lib/localAttributionsTypes";
    import { formatNodeKeyForDisplay } from "../lib/localAttributionsTypes";
    import { colors, getEdgeColor, getOutputNodeColor } from "../lib/colors";
    import { lerp, calcTooltipPos, sortComponentsByImportance, computeComponentOffsets } from "./local-attr/graphUtils";
    import NodeTooltip from "./local-attr/NodeTooltip.svelte";

    // Constants
    const COMPONENT_SIZE = 8;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;

    // Row order for layout (qkv share a row, lm_head before output)
    const ROW_ORDER = ["wte", "qkv", "o_proj", "c_fc", "down_proj", "lm_head", "output"];
    const QKV_SUBTYPES = ["q_proj", "k_proj", "v_proj"];

    type Props = {
        data: GraphData;
        topK: number;
        componentGap: number;
        layerGap: number;
        hideUnconnectedEdges: boolean;
        activationContextsSummary: ActivationContextsSummary | null;
        stagedNodes: PinnedNode[];
        componentDetailsCache: Record<string, ComponentDetail>;
        componentDetailsLoading: Record<string, boolean>;
        onStagedNodesChange: (nodes: PinnedNode[]) => void;
        onLoadComponentDetail: (layer: string, cIdx: number) => void;
        onEdgeCountChange?: (count: number) => void;
    };

    let {
        data,
        topK,
        componentGap,
        layerGap,
        hideUnconnectedEdges,
        activationContextsSummary,
        stagedNodes,
        componentDetailsCache,
        componentDetailsLoading,
        onStagedNodesChange,
        onLoadComponentDetail,
        onEdgeCountChange,
    }: Props = $props();

    // UI state
    let hoveredNode = $state<HoveredNode | null>(null);
    let hoveredEdge = $state<HoveredEdge | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state({ x: 0, y: 0 });
    let edgeTooltipPos = $state({ x: 0, y: 0 });

    // Refs
    let graphContainer: HTMLDivElement;

    // Parse layer name into structured info
    function parseLayer(name: string): LayerInfo {
        if (name === "wte") {
            return { name, block: -1, type: "embed", subtype: "wte" };
        }
        if (name === "lm_head") {
            return { name, block: Infinity - 1, type: "mlp", subtype: "lm_head" };
        }
        if (name === "output") {
            return { name, block: Infinity, type: "output", subtype: "output" };
        }
        const m = name.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
        if (!m) throw new Error(`parseLayer: unrecognized layer name: ${name}`);
        return { name, block: +m[1], type: m[2] as "attn" | "mlp", subtype: m[3] };
    }

    function getRowKey(layer: string): string {
        const info = parseLayer(layer);
        if (QKV_SUBTYPES.includes(info.subtype)) {
            return `h.${info.block}.qkv`;
        }
        return layer;
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

    // All nodes from nodeCiVals (for layout and rendering)
    const allNodes = $derived(new SvelteSet(Object.keys(data.nodeCiVals)));

    // Filter edges by topK (for rendering)
    const filteredEdges = $derived.by(() => {
        const edgesCopy = [...data.edges];
        const sortedEdges = edgesCopy.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
        return sortedEdges.slice(0, topK);
    });

    // Build layout
    const { nodePositions, layerYPositions, seqWidths, seqXStarts, width, height } = $derived.by(() => {
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
        const parseRow = (r: string) => {
            if (r === "wte") return { block: -1, subtype: "wte" };
            if (r === "lm_head") return { block: Infinity - 1, subtype: "lm_head" };
            if (r === "output") return { block: Infinity, subtype: "output" };
            const mQkv = r.match(/h\.(\d+)\.qkv/);
            if (mQkv) return { block: +mQkv[1], subtype: "qkv" };
            const m = r.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
            if (!m) throw new Error(`parseRow: unrecognized row key: ${r}`);
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

        // Assign Y positions to rows
        const rowYPositions: Record<string, number> = {};
        let currentY = MARGIN.top;
        for (const row of rows.slice().reverse()) {
            rowYPositions[row] = currentY;
            currentY += COMPONENT_SIZE + layerGap;
        }

        // Map each layer to its row's Y position
        const layerYPositions: Record<string, number> = {};
        for (const layer of allLayers) {
            const rowKey = getRowKey(layer);
            layerYPositions[layer] = rowYPositions[rowKey];
        }

        // Calculate column widths
        const tokens = data.tokens;
        const maxComponentsPerSeq = tokens.map((_, seqIdx) => {
            let maxAtSeq = 0;
            for (const row of rows) {
                if (row.endsWith(".qkv")) {
                    const blockMatch = row.match(/h\.(\d+)/);
                    if (blockMatch) {
                        const block = blockMatch[1];
                        let totalQkv = 0;
                        for (const subtype of QKV_SUBTYPES) {
                            const layer = `h.${block}.attn.${subtype}`;
                            const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                            totalQkv += nodes.length;
                        }
                        totalQkv += 2; // gaps between groups
                        maxAtSeq = Math.max(maxAtSeq, totalQkv);
                    }
                } else {
                    for (const layer of allLayers) {
                        if (getRowKey(layer) === row) {
                            const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                            maxAtSeq = Math.max(maxAtSeq, nodes.length);
                        }
                    }
                }
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

        // Position nodes
        const nodePositions: Record<string, NodePosition> = {};
        const QKV_GROUP_GAP = COMPONENT_SIZE + componentGap;

        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const isQkv = QKV_SUBTYPES.includes(info.subtype);

            for (let seqIdx = 0; seqIdx < tokens.length; seqIdx++) {
                const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!nodes) continue;

                let baseX = seqXStarts[seqIdx] + COL_PADDING;
                const baseY = layerYPositions[layer];

                // For qkv layers, offset X based on subtype
                if (isQkv) {
                    const subtypeIdx = QKV_SUBTYPES.indexOf(info.subtype);
                    for (let i = 0; i < subtypeIdx; i++) {
                        const prevLayer = `h.${info.block}.attn.${QKV_SUBTYPES[i]}`;
                        const prevLayerNodes = nodesPerLayerSeq[`${prevLayer}:${seqIdx}`];
                        const prevCount = prevLayerNodes?.length ?? 0;
                        baseX += prevCount * (COMPONENT_SIZE + componentGap);
                        baseX += QKV_GROUP_GAP;
                    }
                }

                const sorted = sortComponentsByImportance(nodes, layer, seqIdx, data.nodeCiVals, data.outputProbs);
                const offsets = computeComponentOffsets(sorted, COMPONENT_SIZE, componentGap);

                for (const cIdx of nodes) {
                    nodePositions[`${layer}:${seqIdx}:${cIdx}`] = {
                        x: baseX + offsets[cIdx] + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                }
            }
        }

        const totalSeqWidth = seqXStarts[seqXStarts.length - 1] + seqWidths[seqWidths.length - 1];
        const widthVal = totalSeqWidth + MARGIN.right;
        const maxY = Math.max(...Object.values(layerYPositions), 0) + COMPONENT_SIZE;
        const heightVal = maxY + MARGIN.bottom;

        return { nodePositions, layerYPositions, seqWidths, seqXStarts, width: widthVal, height: heightVal };
    });

    const EDGE_HIT_AREA_WIDTH = 4; // Wider invisible stroke for easier hover

    // Check if an edge is connected to any pinned node (exact match including seqIdx)
    function isEdgeConnectedToPinnedNode(src: string, tgt: string): boolean {
        if (stagedNodes.length === 0) return false;
        const [srcLayer, srcSeqIdx, srcCIdx] = src.split(":");
        const [tgtLayer, tgtSeqIdx, tgtCIdx] = tgt.split(":");
        for (const pinned of stagedNodes) {
            if (
                (srcLayer === pinned.layer && +srcSeqIdx === pinned.seqIdx && +srcCIdx === pinned.cIdx) ||
                (tgtLayer === pinned.layer && +tgtSeqIdx === pinned.seqIdx && +tgtCIdx === pinned.cIdx)
            ) {
                return true;
            }
        }
        return false;
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
            if (!isEdgeConnectedToPinnedNode(edge.src, edge.tgt)) continue;

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

    function isNodePinned(layer: string, seqIdx: number, cIdx: number): boolean {
        return stagedNodes.some((p) => p.layer === layer && p.seqIdx === seqIdx && p.cIdx === cIdx);
    }

    // Check if a node key should be highlighted
    // - Pinned nodes: exact match (layer + seqIdx + cIdx)
    // - Hovered: highlight all nodes with same component (layer + cIdx) across all positions
    function isKeyHighlighted(key: string): boolean {
        const [layer, seqIdx, cIdx] = key.split(":");
        // Exact match for pinned nodes
        if (stagedNodes.some((p) => p.layer === layer && p.seqIdx === +seqIdx && p.cIdx === +cIdx)) {
            return true;
        }
        // For hover: highlight all nodes with same component (across all positions)
        if (hoveredNode && !isNodePinned(hoveredNode.layer, hoveredNode.seqIdx, hoveredNode.cIdx)) {
            if (layer === hoveredNode.layer && +cIdx === hoveredNode.cIdx) {
                return true;
            }
        }
        return false;
    }

    // Set of highlighted node keys for edge highlighting
    const highlightedKeys = $derived.by(() => {
        const keys = new SvelteSet<string>();
        for (const nodeKey of Object.keys(nodePositions)) {
            if (isKeyHighlighted(nodeKey)) {
                keys.add(nodeKey);
            }
        }
        return keys;
    });

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
                    fill = getOutputNodeColor(probEntry.prob);
                    opacity = 0.4 + probEntry.prob * 0.6;
                }
            } else {
                // Component nodes: opacity based on CI (brighter = higher CI)
                const ci = data.nodeCiVals[`${layer}:${seqIdx}:${cIdx}`] || 0;
                const intensity = Math.min(1, ci / maxCi);
                opacity = 0.2 + intensity * 0.8;
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
        tooltipPos = calcTooltipPos(event.clientX, event.clientY);

        // Lazy load component details if needed
        if (layer !== "output" && activationContextsSummary) {
            onLoadComponentDetail(layer, cIdx);
        }
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
        }, 50);
    }

    function handleNodeClick(layer: string, seqIdx: number, cIdx: number) {
        const idx = stagedNodes.findIndex((p) => p.layer === layer && p.seqIdx === seqIdx && p.cIdx === cIdx);
        if (idx >= 0) {
            onStagedNodesChange(stagedNodes.filter((_, i) => i !== idx));
        } else {
            onStagedNodesChange([...stagedNodes, { layer, seqIdx, cIdx }]);
        }
        hoveredNode = null;
    }

    function pinComponent(layer: string, cIdx: number, seqIdx: number) {
        const alreadyPinned = stagedNodes.some((p) => p.layer === layer && p.cIdx === cIdx && p.seqIdx === seqIdx);
        if (alreadyPinned) return;
        onStagedNodesChange([...stagedNodes, { layer, cIdx, seqIdx }]);
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

    // Track previously highlighted/dimmed/hidden edges to minimize DOM updates
    let prevHighlightedEdges = new SvelteSet<Element>();
    let prevDimmedEdges = new SvelteSet<Element>();
    let prevHiddenEdges = new SvelteSet<Element>();

    // Update edge highlighting and visibility via $effect (DOM manipulation for performance)
    // Only updates edges that actually changed state
    $effect(() => {
        if (!graphContainer) {
            return;
        }

        const currentHighlighted = new SvelteSet<Element>();
        const currentDimmed = new SvelteSet<Element>();
        const currentHidden = new SvelteSet<Element>();
        // Only target visible edges for highlighting (not hit areas)
        const edges = graphContainer.querySelectorAll(".edge-visible");
        const hasSelection = highlightedKeys.size > 0;
        const shouldHideUnconnected = hideUnconnectedEdges && hasSelection;

        // Build set of currently highlighted, dimmed, and hidden edges
        edges.forEach((el) => {
            const src = el.getAttribute("data-src") || "";
            const tgt = el.getAttribute("data-tgt") || "";
            const isConnectedToPinned = highlightedKeys.has(src) || highlightedKeys.has(tgt);

            if (isConnectedToPinned) {
                // Check if this is the hovered edge
                if (hoveredEdge && hoveredEdge.src === src && hoveredEdge.tgt === tgt) {
                    currentHighlighted.add(el);
                } else if (hoveredEdge) {
                    // Another edge is being hovered, dim this one
                    currentDimmed.add(el);
                } else {
                    // No edge hovered, highlight all pinned-connected edges
                    currentHighlighted.add(el);
                }
            } else if (shouldHideUnconnected) {
                // Hide edges not connected to any selected node
                currentHidden.add(el);
            }
        });

        // Remove highlight from edges no longer highlighted
        for (const el of prevHighlightedEdges) {
            if (!currentHighlighted.has(el)) {
                el.classList.remove("highlighted");
            }
        }

        // Add highlight to newly highlighted edges
        for (const el of currentHighlighted) {
            if (!prevHighlightedEdges.has(el)) {
                el.classList.add("highlighted");
            }
        }

        // Remove dimmed from edges no longer dimmed
        for (const el of prevDimmedEdges) {
            if (!currentDimmed.has(el)) {
                el.classList.remove("dimmed");
            }
        }

        // Add dimmed to newly dimmed edges
        for (const el of currentDimmed) {
            if (!prevDimmedEdges.has(el)) {
                el.classList.add("dimmed");
            }
        }

        // Remove hidden class from edges no longer hidden
        for (const el of prevHiddenEdges) {
            if (!currentHidden.has(el)) {
                el.classList.remove("hidden");
            }
        }

        // Add hidden class to newly hidden edges
        for (const el of currentHidden) {
            if (!prevHiddenEdges.has(el)) {
                el.classList.add("hidden");
            }
        }

        prevHighlightedEdges = currentHighlighted;
        prevDimmedEdges = currentDimmed;
        prevHiddenEdges = currentHidden;
    });

    // Notify parent of edge count changes
    $effect(() => {
        onEdgeCountChange?.(filteredEdges.length);
    });
</script>

<div class="graph-wrapper" bind:this={graphContainer}>
    <div class="layer-labels-container" style="width: {LABEL_WIDTH}px;">
        <svg width={LABEL_WIDTH} {height} style="display: block;">
            {#each Object.entries(layerYPositions) as [layer, y] (layer)}
                {@const info = parseLayer(layer)}
                {@const yCenter = y + COMPONENT_SIZE / 2}
                {@const rowKey = getRowKey(layer)}
                {@const label = rowKey.endsWith(".qkv")
                    ? `${info.block}.q/k/v`
                    : layer === "wte" || layer === "output"
                      ? layer
                      : layer === "lm_head"
                        ? "W_U"
                        : `${info.block}.${info.subtype}`}
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
                    {label}
                </text>
            {/each}
        </svg>
    </div>

    <div class="graph-container">
        <!-- svelte-ignore a11y_no_static_element_interactions, a11y_mouse_events_have_key_events -->
        <svg {width} {height} onmouseover={handleEdgeMouseEnter} onmouseout={handleEdgeMouseLeave}>
            <!-- Edges (bulk rendered for performance, uses @html for large SVG performance) -->
            <g class="edges-layer">
                <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                {@html edgesSvgString}
            </g>

            <!-- Nodes (reactive for interactivity) -->
            <g class="nodes-layer">
                {#each Object.entries(nodePositions) as [key, pos] (key)}
                    {@const [layer, seqIdxStr, cIdxStr] = key.split(":")}
                    {@const seqIdx = parseInt(seqIdxStr)}
                    {@const cIdx = parseInt(cIdxStr)}
                    {@const isHighlighted = isKeyHighlighted(key)}
                    {@const style = nodeStyles[key]}
                    <!-- svelte-ignore a11y_click_events_have_key_events -->
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
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
        </svg>

        <div class="token-labels-container">
            <svg {width} height="50" style="display: block;">
                {#each data.tokens as token, i (i)}
                    {@const colCenter = seqXStarts[i] + seqWidths[i] / 2}
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
            </svg>
        </div>
    </div>

    <!-- Edge tooltip -->
    {#if hoveredEdge}
        <div class="edge-tooltip" style="left: {edgeTooltipPos.x}px; top: {edgeTooltipPos.y}px;">
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Src</span>
                <code>{formatNodeKeyForDisplay(hoveredEdge.src)}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Tgt</span>
                <code>{formatNodeKeyForDisplay(hoveredEdge.tgt)}</code>
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
    {#if hoveredNode && !isNodePinned(hoveredNode.layer, hoveredNode.seqIdx, hoveredNode.cIdx)}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            {activationContextsSummary}
            {componentDetailsCache}
            {componentDetailsLoading}
            outputProbs={data.outputProbs}
            nodeCiVals={data.nodeCiVals}
            tokens={data.tokens}
            edges={data.edges}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
            onPinComponent={pinComponent}
        />
    {/if}
</div>

<style>
    .graph-wrapper {
        display: flex;
        border: 1px solid var(--border-default);
        background: var(--bg-surface);
        overflow: hidden;
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

    :global(.edge.dimmed) {
        opacity: 0.15 !important;
    }

    :global(.edge.hidden) {
        display: none;
    }

    .node-group {
        cursor: pointer;
    }

    .node.highlighted {
        stroke: var(--accent-primary) !important;
        stroke-width: 2px !important;
        filter: brightness(1.2);
        opacity: 1 !important;
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
