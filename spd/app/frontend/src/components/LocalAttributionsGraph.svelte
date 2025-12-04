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
    } from "../lib/localAttributionsTypes";
    import * as api from "../lib/localAttributionsApi";
    import type { ComponentDetail } from "../lib/localAttributionsTypes";

    // Constants
    const COMPONENT_SIZE = 8;
    const COMPONENT_GAP = 2;
    const LAYER_GAP = 30;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;
    const NODE_COLOR = "#333";

    // Row order for layout (qkv share a row)
    const ROW_ORDER = ["wte", "qkv", "o_proj", "c_fc", "down_proj", "output"];
    const QKV_SUBTYPES = ["q_proj", "k_proj", "v_proj"];

    type Props = {
        data: GraphData;
        topK: number;
        nodeLayout: "importance" | "shuffled" | "jittered";
        activationContextsSummary: ActivationContextsSummary | null;
        pinnedNodes: PinnedNode[];
        onPinnedNodesChange: (nodes: PinnedNode[]) => void;
    };

    let { data, topK, nodeLayout, activationContextsSummary, pinnedNodes, onPinnedNodesChange }: Props = $props();

    // UI state
    let hoveredNode = $state<HoveredNode | null>(null);
    let hoveredEdge = $state<HoveredEdge | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state({ x: 0, y: 0 });
    let edgeTooltipPos = $state({ x: 0, y: 0 });

    // Component details cache (lazy-loaded)
    let componentDetailsCache = $state<Record<string, ComponentDetail>>({});
    let componentDetailsLoading = $state<Record<string, boolean>>({});

    // Refs
    let graphContainer: HTMLDivElement;

    // Parse layer name into structured info
    function parseLayer(name: string): LayerInfo {
        if (name === "wte") {
            return { name, block: -1, type: "embed", subtype: "wte" };
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

    // Compute importance maps from edges
    const { componentImportanceLocal, maxImportanceLocal, maxAbsAttr } = $derived.by(() => {
        const componentImportanceLocal: Record<string, number> = {};
        let maxAbsAttr = 1;

        for (const edge of data.edges) {
            const valSq = edge.val * edge.val;
            const absVal = Math.abs(edge.val);
            if (absVal > maxAbsAttr) maxAbsAttr = absVal;

            componentImportanceLocal[edge.src] = (componentImportanceLocal[edge.src] || 0) + valSq;
            componentImportanceLocal[edge.tgt] = (componentImportanceLocal[edge.tgt] || 0) + valSq;
        }

        let maxImportanceLocal = 1;
        for (const imp of Object.values(componentImportanceLocal)) {
            if (imp > maxImportanceLocal) maxImportanceLocal = imp;
        }

        return { componentImportanceLocal, maxImportanceLocal, maxAbsAttr };
    });

    // Filter edges by topK and build active nodes set
    const { filteredEdges, activeNodes } = $derived.by(() => {
        const edgesCopy = [...data.edges];

        const sortedEdges = edgesCopy.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));

        const filteredEdges = sortedEdges.slice(0, topK);

        const activeNodes = new SvelteSet<string>();
        for (const edge of filteredEdges) {
            activeNodes.add(edge.src);
            activeNodes.add(edge.tgt);
        }

        // For output nodes: include all tokens with prob >= min prob of kept tokens
        const outputNodesWithEdges = new SvelteSet<string>();
        for (const nodeKey of activeNodes) {
            if (nodeKey.startsWith("output:")) {
                outputNodesWithEdges.add(nodeKey);
            }
        }

        if (outputNodesWithEdges.size > 0) {
            let minProb = Infinity;
            for (const nodeKey of outputNodesWithEdges) {
                const [, seqIdx, cIdx] = nodeKey.split(":");
                const probKey = `${seqIdx}:${cIdx}`;
                const entry = data.outputProbs[probKey];
                if (entry && entry.prob < minProb) {
                    minProb = entry.prob;
                }
            }

            const outputProbsPlain = $state.snapshot(data.outputProbs);
            for (const [probKey, entry] of Object.entries(outputProbsPlain)) {
                if (entry.prob >= minProb) {
                    const [seqIdx, cIdx] = probKey.split(":");
                    activeNodes.add(`output:${seqIdx}:${cIdx}`);
                }
            }
        }

        return { filteredEdges, activeNodes };
    });

    // Build layout
    const { nodePositions, layerYPositions, seqWidths, seqXStarts, width, height } = $derived.by(() => {
        const nodesPerLayerSeq: Record<string, number[]> = {};
        const allLayers = new SvelteSet<string>();
        const allRows = new SvelteSet<string>();

        for (const nodeKey of activeNodes) {
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
            currentY += COMPONENT_SIZE + LAYER_GAP;
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
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + COMPONENT_GAP) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Position nodes
        const nodePositions: Record<string, NodePosition> = {};
        const QKV_GROUP_GAP = COMPONENT_SIZE + COMPONENT_GAP;

        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const isQkv = QKV_SUBTYPES.includes(info.subtype);
            const isOutput = layer === "output";

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
                        baseX += prevCount * (COMPONENT_SIZE + COMPONENT_GAP);
                        baseX += QKV_GROUP_GAP;
                    }
                }

                const cellWidth = seqWidths[seqIdx] - COL_PADDING * 2;
                const offsets = getComponentOffsets(nodes, layer, seqIdx, isOutput, cellWidth);

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

    // Get component offsets based on layout strategy
    function getComponentOffsets(
        components: number[],
        layer: string,
        seqIdx: number,
        isOutput: boolean,
        cellWidth: number,
    ): Record<number, number> {
        const n = components.length;
        const offsets: Record<number, number> = {};

        if (nodeLayout === "importance") {
            const sorted = [...components].sort((a, b) => {
                if (isOutput) {
                    const entryA = data.outputProbs[`${seqIdx}:${a}`];
                    const entryB = data.outputProbs[`${seqIdx}:${b}`];
                    return (entryB?.prob ?? 0) - (entryA?.prob ?? 0);
                }
                const impA = componentImportanceLocal[`${layer}:${seqIdx}:${a}`] ?? 0;
                const impB = componentImportanceLocal[`${layer}:${seqIdx}:${b}`] ?? 0;
                return impB - impA;
            });
            for (let i = 0; i < n; i++) {
                offsets[sorted[i]] = i * (COMPONENT_SIZE + COMPONENT_GAP);
            }
            return offsets;
        }

        if (nodeLayout === "shuffled") {
            const seed = hashString(`${layer}:${seqIdx}`);
            const shuffled = seededShuffle([...components], seed);
            for (let i = 0; i < n; i++) {
                offsets[shuffled[i]] = i * (COMPONENT_SIZE + COMPONENT_GAP);
            }
            return offsets;
        }

        // jittered
        const seed = hashString(`${layer}:${seqIdx}`);
        const shuffled = seededShuffle([...components], seed);

        let jitterSeed = seed;
        const jitterRandom = () => {
            jitterSeed |= 0;
            jitterSeed = (jitterSeed + 0x6d2b79f5) | 0;
            let t = Math.imul(jitterSeed ^ (jitterSeed >>> 15), 1 | jitterSeed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };

        const totalSpace = cellWidth - n * COMPONENT_SIZE;
        const gap = totalSpace / (n + 1);
        const maxJitter = gap / 2;

        for (let i = 0; i < n; i++) {
            const baseOffset = gap + i * (COMPONENT_SIZE + gap);
            const jitter = (jitterRandom() - 0.5) * 2 * maxJitter;
            const jitteredOffset = Math.max(0, Math.min(cellWidth - COMPONENT_SIZE, baseOffset + jitter));
            offsets[shuffled[i]] = jitteredOffset;
        }
        return offsets;
    }

    function hashString(str: string): number {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = (hash << 5) - hash + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    function seededShuffle<T>(arr: T[], seed: number): T[] {
        const random = () => {
            seed |= 0;
            seed = (seed + 0x6d2b79f5) | 0;
            let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };

        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    function lerp(min: number, max: number, t: number): number {
        return min + (max - min) * t;
    }

    // Build SVG edges string (for {@html} - performance optimization)
    const edgesSvgString = $derived.by(() => {
        let svg = "";
        for (const edge of filteredEdges) {
            const p1 = nodePositions[edge.src];
            const p2 = nodePositions[edge.tgt];
            if (p1 && p2) {
                const color = edge.val > 0 ? "#2196f3" : "#f44336";
                const w = lerp(1, 4, Math.abs(edge.val) / maxAbsAttr);
                const op = lerp(0, 0.5, Math.abs(edge.val) / maxAbsAttr);
                const dy = Math.abs(p2.y - p1.y);
                const curveOffset = Math.max(20, dy * 0.4);
                const cp1y = p1.y - curveOffset;
                const cp2y = p2.y + curveOffset;
                const d = `M ${p1.x},${p1.y} C ${p1.x},${cp1y} ${p2.x},${cp2y} ${p2.x},${p2.y}`;
                svg += `<path class="edge" data-src="${edge.src}" data-tgt="${edge.tgt}" data-val="${edge.val}" d="${d}" stroke="${color}" stroke-width="${w}" opacity="${op}" fill="none"/>`;
            }
        }
        return svg;
    });

    // Pinned node keys (stable - only changes when user clicks to pin/unpin)
    const pinnedKeys = $derived.by(() => {
        const keys = new SvelteSet<string>();
        for (const pinned of pinnedNodes) {
            for (const nodeKey of Object.keys(nodePositions)) {
                const [layer, , cIdx] = nodeKey.split(":");
                if (layer === pinned.layer && +cIdx === pinned.cIdx) {
                    keys.add(nodeKey);
                }
            }
        }
        return keys;
    });

    // Hovered node keys (changes frequently but kept separate to minimize re-renders)
    const hoveredKeys = $derived.by(() => {
        if (!hoveredNode || isNodePinned(hoveredNode.layer, hoveredNode.cIdx)) {
            return new SvelteSet<string>();
        }
        const keys = new SvelteSet<string>();
        for (const nodeKey of Object.keys(nodePositions)) {
            const [layer, , cIdx] = nodeKey.split(":");
            if (layer === hoveredNode.layer && +cIdx === hoveredNode.cIdx) {
                keys.add(nodeKey);
            }
        }
        return keys;
    });

    // Helper to check if a key is highlighted (avoids creating new Set on every hover)
    function isKeyHighlighted(key: string): boolean {
        return pinnedKeys.has(key) || hoveredKeys.has(key);
    }

    // Combined set for edge highlighting effect only
    const highlightedKeys = $derived(new SvelteSet([...pinnedKeys, ...hoveredKeys]));

    function isNodePinned(layer: string, cIdx: number): boolean {
        return pinnedNodes.some((p) => p.layer === layer && p.cIdx === cIdx);
    }

    // Pre-compute node styles (fill, opacity) - only recomputes when data/layout changes, not on hover
    const nodeStyles = $derived.by(() => {
        const styles: Record<string, { fill: string; opacity: number }> = {};

        for (const nodeKey of Object.keys(nodePositions)) {
            const [layer, seqIdxStr, cIdxStr] = nodeKey.split(":");
            const seqIdx = parseInt(seqIdxStr);
            const cIdx = parseInt(cIdxStr);

            let fill = NODE_COLOR;
            let opacity = 0.2;

            if (layer === "output") {
                const probEntry = data.outputProbs[`${seqIdx}:${cIdx}`];
                if (probEntry) {
                    const prob = probEntry.prob;
                    const saturation = 20 + prob * 60;
                    const lightness = 70 - prob * 35;
                    fill = `hsl(120, ${saturation}%, ${lightness}%)`;
                    opacity = 0.4 + prob * 0.6;
                }
            } else {
                const importance = componentImportanceLocal[`${layer}:${seqIdx}:${cIdx}`] || 0;
                const intensity = Math.min(1, importance / maxImportanceLocal);
                opacity = 0.2 + intensity * 0.8;
            }

            styles[nodeKey] = { fill, opacity };
        }

        return styles;
    });

    // Event handlers
    function handleNodeMouseEnter(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        hoveredNode = { layer, seqIdx, cIdx };
        tooltipPos = calcTooltipPos(event.clientX, event.clientY);

        // Lazy load component details if needed
        if (layer !== "output" && !activationContextsSummary) {
            // No summary available
        } else if (layer !== "output") {
            loadComponentDetailIfNeeded(layer, cIdx);
        }
    }

    function handleNodeMouseLeave() {
        setTimeout(() => {
            if (!isHoveringTooltip) {
                hoveredNode = null;
            }
        }, 100);
    }

    function handleNodeClick(layer: string, cIdx: number) {
        const idx = pinnedNodes.findIndex((p) => p.layer === layer && p.cIdx === cIdx);
        if (idx >= 0) {
            onPinnedNodesChange(pinnedNodes.filter((_, i) => i !== idx));
        } else {
            onPinnedNodesChange([...pinnedNodes, { layer, cIdx }]);
        }
        hoveredNode = null;
    }

    function handleEdgeMouseEnter(event: MouseEvent) {
        const target = event.target as SVGElement;
        if (target.classList.contains("edge")) {
            const src = target.getAttribute("data-src") || "";
            const tgt = target.getAttribute("data-tgt") || "";
            const val = parseFloat(target.getAttribute("data-val") || "0");
            hoveredEdge = { src, tgt, val };
            edgeTooltipPos = { x: event.clientX + 10, y: event.clientY + 10 };
        }
    }

    function handleEdgeMouseLeave(event: MouseEvent) {
        const target = event.target as SVGElement;
        if (target.classList.contains("edge")) {
            hoveredEdge = null;
        }
    }

    function calcTooltipPos(mouseX: number, mouseY: number) {
        const padding = 15;
        let left = mouseX + padding;
        let top = mouseY + padding;
        if (typeof window !== "undefined") {
            if (left + 500 > window.innerWidth) left = mouseX - 500 - padding;
            if (top + 400 > window.innerHeight) top = mouseY - 400 - padding;
        }
        return { x: Math.max(0, left), y: Math.max(0, top) };
    }

    async function loadComponentDetailIfNeeded(layer: string, cIdx: number) {
        const cacheKey = `${layer}:${cIdx}`;
        if (componentDetailsCache[cacheKey] || componentDetailsLoading[cacheKey]) return;

        componentDetailsLoading[cacheKey] = true;
        try {
            const detail = await api.getComponentDetail(layer, cIdx);
            componentDetailsCache[cacheKey] = detail;
        } catch (e) {
            console.error(`Failed to load component detail for ${cacheKey}:`, e);
        } finally {
            componentDetailsLoading[cacheKey] = false;
        }
    }

    function escapeHtml(text: string): string {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    // Track previously highlighted edges to minimize DOM updates
    let prevHighlightedEdges = new SvelteSet<Element>();

    // Update edge highlighting via $effect (DOM manipulation for performance)
    // Only updates edges that actually changed state
    $effect(() => {
        if (!graphContainer) {
            return;
        }

        const currentHighlighted = new SvelteSet<Element>();
        const edges = graphContainer.querySelectorAll(".edge");

        // Build set of currently highlighted edges
        edges.forEach((el) => {
            const src = el.getAttribute("data-src") || "";
            const tgt = el.getAttribute("data-tgt") || "";
            if (highlightedKeys.has(src) || highlightedKeys.has(tgt)) {
                currentHighlighted.add(el);
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

        prevHighlightedEdges = currentHighlighted;
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
                      : `${info.block}.${info.subtype}`}
                <text
                    x={LABEL_WIDTH - 10}
                    y={yCenter}
                    text-anchor="end"
                    dominant-baseline="middle"
                    font-size="11"
                    font-weight="500"
                    fill={NODE_COLOR}
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
                        onmouseenter={(e) => handleNodeMouseEnter(e, layer, seqIdx, cIdx)}
                        onmouseleave={handleNodeMouseLeave}
                        onclick={() => handleNodeClick(layer, cIdx)}
                    />
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
                        font-size="12"
                        font-family="monospace"
                        font-weight="500"
                    >
                        {token}
                    </text>
                    <text x={colCenter} y="38" text-anchor="middle" font-size="10" fill="#999">[{i}]</text>
                {/each}
            </svg>
        </div>
    </div>

    <!-- Edge tooltip -->
    {#if hoveredEdge}
        <div class="edge-tooltip" style="left: {edgeTooltipPos.x}px; top: {edgeTooltipPos.y}px;">
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">From:</span>
                <code>{hoveredEdge.src}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">To:</span>
                <code>{hoveredEdge.tgt}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Strength:</span>
                <span style="color: {hoveredEdge.val > 0 ? '#2196f3' : '#f44336'}; font-weight: bold;">
                    {hoveredEdge.val.toFixed(4)}
                </span>
            </div>
        </div>
    {/if}

    <!-- Node tooltip -->
    {#if hoveredNode && !isNodePinned(hoveredNode.layer, hoveredNode.cIdx)}
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
            <h3>{hoveredNode.layer}:{hoveredNode.cIdx}</h3>

            {#if hoveredNode.layer === "output"}
                {@const probEntry = data.outputProbs[`${hoveredNode.seqIdx}:${hoveredNode.cIdx}`]}
                {#if probEntry}
                    <div
                        class="output-header"
                        style="background: linear-gradient(90deg, rgba(76, 175, 80, {Math.min(
                            0.8,
                            probEntry.prob + 0.1,
                        )}) 0%, rgba(76, 175, 80, 0.1) 100%);"
                    >
                        <div class="output-token">"{escapeHtml(probEntry.token)}"</div>
                        <div class="output-prob">{(probEntry.prob * 100).toFixed(1)}% probability</div>
                    </div>
                    <p class="stats">
                        <strong>Position:</strong>
                        {hoveredNode.seqIdx} |
                        <strong>Vocab ID:</strong>
                        {hoveredNode.cIdx}
                    </p>
                {/if}
            {:else}
                {@const summary = activationContextsSummary?.[hoveredNode.layer]?.find(
                    (s) => s.subcomponent_idx === hoveredNode?.cIdx,
                )}
                {@const detail = componentDetailsCache[`${hoveredNode.layer}:${hoveredNode.cIdx}`]}
                {@const isLoading = componentDetailsLoading[`${hoveredNode.layer}:${hoveredNode.cIdx}`]}

                <p class="stats">
                    <strong>Position:</strong>
                    {hoveredNode.seqIdx}
                    {#if summary}
                        | <strong>Mean CI:</strong>
                        {summary.mean_ci.toFixed(4)}
                    {/if}
                </p>

                {#if detail}
                    {#if detail.example_tokens?.length > 0}
                        <h4>Top Activating Examples</h4>
                        {#each detail.example_tokens.slice(0, 5) as tokens, i (i)}
                            {@const ciVals = detail.example_ci[i]}
                            {@const activePos = detail.example_active_pos[i]}
                            <div class="example-row">
                                {#each tokens as token, j (j)}
                                    {@const ci = ciVals[j]}
                                    {@const isActive = j === activePos}
                                    <span
                                        class="example-token"
                                        class:active={isActive}
                                        style="background: rgba(255, 100, 100, {Math.min(1, ci * 5)});"
                                    >
                                        {token}
                                    </span>
                                {/each}
                            </div>
                        {/each}
                    {/if}

                    <div class="tables-row">
                        {#if detail.pr_tokens?.length > 0}
                            <div>
                                <h4>Top Input Tokens</h4>
                                <table class="pr-table">
                                    <tbody>
                                        {#each detail.pr_tokens.slice(0, 10) as token, i (i)}
                                            <tr>
                                                <td><code>{token}</code></td>
                                                <td>{detail.pr_precisions[i]?.toFixed(3)}</td>
                                            </tr>
                                        {/each}
                                    </tbody>
                                </table>
                            </div>
                        {/if}

                        {#if detail.predicted_tokens?.length}
                            <div>
                                <h4>Top Predicted</h4>
                                <table class="pr-table">
                                    <tbody>
                                        {#each detail.predicted_tokens.slice(0, 10) as token, i (i)}
                                            <tr>
                                                <td><code>{token}</code></td>
                                                <td>{detail.predicted_probs?.[i]?.toFixed(3)}</td>
                                            </tr>
                                        {/each}
                                    </tbody>
                                </table>
                            </div>
                        {/if}
                    </div>
                {:else if isLoading}
                    <p class="loading-text">Loading details...</p>
                {/if}
            {/if}
        </div>
    {/if}
</div>

<!-- Pinned components panel -->
{#if pinnedNodes.length > 0}
    <div class="pinned-container">
        <h3>
            <span>Pinned Components ({pinnedNodes.length})</span>
            <button onclick={() => onPinnedNodesChange([])}>Clear all</button>
        </h3>
        <div class="pinned-items">
            {#each pinnedNodes as pinned (`${pinned.layer}:${pinned.cIdx}`)}
                {@const detail = componentDetailsCache[`${pinned.layer}:${pinned.cIdx}`]}
                <div class="pinned-item">
                    <div class="pinned-header">
                        <strong>{pinned.layer}:{pinned.cIdx}</strong>
                        <button
                            class="unpin-btn"
                            onclick={() => onPinnedNodesChange(pinnedNodes.filter((p) => p !== pinned))}
                        >
                            ✕
                        </button>
                    </div>

                    {#if pinned.layer === "output"}
                        <!-- Find all positions where this token appears -->
                        {@const positions = Object.entries(data.outputProbs)
                            .filter(([key]) => key.endsWith(`:${pinned.cIdx}`))
                            .map(([key, entry]) => ({
                                seqIdx: parseInt(key.split(":")[0]),
                                prob: entry.prob,
                                token: entry.token,
                            }))
                            .sort((a, b) => b.prob - a.prob)}
                        {#if positions.length > 0}
                            <p><strong>"{positions[0].token}"</strong></p>
                            <table class="pr-table">
                                <tbody>
                                    {#each positions as pos (pos.seqIdx)}
                                        <tr>
                                            <td>Pos {pos.seqIdx}</td>
                                            <td>{(pos.prob * 100).toFixed(2)}%</td>
                                        </tr>
                                    {/each}
                                </tbody>
                            </table>
                        {/if}
                    {:else if detail}
                        {#if detail.example_tokens?.length > 0}
                            <h4>Examples</h4>
                            {#each detail.example_tokens.slice(0, 3) as tokens, i (i)}
                                {@const ciVals = detail.example_ci[i]}
                                {@const activePos = detail.example_active_pos[i]}
                                <div class="example-row">
                                    {#each tokens as token, j (j)}
                                        {@const ci = ciVals[j]}
                                        {@const isActive = j === activePos}
                                        <span
                                            class="example-token"
                                            class:active={isActive}
                                            style="background: rgba(255, 100, 100, {Math.min(1, ci * 5)});"
                                        >
                                            {token}
                                        </span>
                                    {/each}
                                </div>
                            {/each}
                        {/if}
                    {:else}
                        <p class="loading-text">Loading...</p>
                    {/if}
                </div>
            {/each}
        </div>
    </div>
{/if}

<div class="legend">
    <span class="edge-count">Showing {filteredEdges.length} edges</span>
    <span class="legend-item">
        <span class="edge-pos"></span> Positive
    </span>
    <span class="legend-item">
        <span class="edge-neg"></span> Negative
    </span>
</div>

<style>
    .graph-wrapper {
        display: flex;
        max-height: 70vh;
        background: white;
        overflow: hidden;
    }

    .layer-labels-container {
        position: sticky;
        left: 0;
        background: white;
        border-right: 1px solid #eee;
        z-index: 11;
        flex-shrink: 0;
    }

    .graph-container {
        overflow: auto;
        flex: 1;
        position: relative;
    }

    .token-labels-container {
        position: sticky;
        bottom: 0;
        background: white;
        border-top: 1px solid #eee;
        z-index: 10;
    }

    svg {
        display: block;
    }

    :global(.edge) {
        transition:
            opacity 0.15s,
            stroke-width 0.15s;
    }

    :global(.edge.highlighted) {
        opacity: 1 !important;
        stroke-width: 3 !important;
    }

    .node {
        transition:
            stroke-width 0.15s,
            filter 0.15s;
        cursor: pointer;
    }

    .node.highlighted {
        stroke: #000 !important;
        stroke-width: 2.5 !important;
        filter: brightness(0.7) saturate(1.5);
        opacity: 1 !important;
    }

    .edge-tooltip,
    .node-tooltip {
        position: fixed;
        padding: 0.75rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        pointer-events: auto;
    }

    .edge-tooltip {
        font-size: 0.85rem;
    }

    .edge-tooltip-row {
        margin: 0.2rem 0;
    }

    .edge-tooltip-label {
        color: #666;
        margin-right: 0.5rem;
    }

    .node-tooltip {
        max-width: 600px;
        max-height: 500px;
        overflow-y: auto;
    }

    .node-tooltip h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
    }

    .node-tooltip h4 {
        margin: 0.75rem 0 0.25rem 0;
        font-size: 0.85rem;
        color: #333;
    }

    .output-header {
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }

    .output-token {
        font-size: 1.2em;
        font-weight: bold;
    }

    .output-prob {
        font-size: 1em;
        color: #333;
    }

    .stats {
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }

    .example-row {
        display: flex;
        flex-wrap: wrap;
        gap: 2px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }

    .example-token {
        padding: 2px 4px;
        border-radius: 2px;
    }

    .example-token.active {
        font-weight: bold;
        border: 1px solid #333;
    }

    .tables-row {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
    }

    .pr-table {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .pr-table td {
        padding: 0.15rem 0.5rem;
    }

    .loading-text {
        font-size: 0.85rem;
        color: #666;
    }

    .pinned-container {
        margin-top: 1rem;
        padding: 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .pinned-container h3 {
        margin: 0 0 0.75rem 0;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .pinned-container h3 button {
        font-size: 0.8rem;
        padding: 0.25rem 0.5rem;
        cursor: pointer;
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .pinned-items {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }

    .pinned-item {
        flex-shrink: 0;
        min-width: 300px;
        max-width: 400px;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.75rem;
        background: #fafafa;
    }

    .pinned-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .unpin-btn {
        cursor: pointer;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
    }

    .pinned-item h4 {
        margin: 0.5rem 0 0.25rem 0;
        font-size: 0.8rem;
    }

    .legend {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #666;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .edge-pos,
    .edge-neg {
        display: inline-block;
        width: 24px;
        height: 3px;
    }

    .edge-pos {
        background: #2196f3;
    }

    .edge-neg {
        background: #f44336;
    }

    .edge-count {
        font-weight: 500;
    }
</style>
