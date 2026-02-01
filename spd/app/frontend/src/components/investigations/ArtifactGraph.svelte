<script lang="ts">
    /**
     * Simplified graph component for rendering artifacts in research reports.
     * Based on PromptAttributionsGraph but without:
     * - Cluster features and runState context
     * - Node tooltips with component lookups
     * - Pinned nodes / staging
     * Keeps: nodes, edges, zoom/pan, basic layout
     */

    import { SvelteSet } from "svelte/reactivity";
    import type { NodePosition, OutputProbability } from "../../lib/promptAttributionsTypes";
    import { getAliasedRowLabel } from "../../lib/layerAliasing";
    import { colors, getEdgeColor, rgbToCss } from "../../lib/colors";
    import { lerp, sortComponentsByImportance, computeComponentOffsets } from "../prompt-attr/graphUtils";
    import { useZoomPan } from "../../lib/useZoomPan.svelte";
    import ZoomControls from "../../lib/ZoomControls.svelte";

    // Constants
    const COMPONENT_SIZE = 8;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;

    // Row order for layout (qkv share a row, lm_head before output)
    const ROW_ORDER = ["wte", "qkv", "o_proj", "c_fc", "down_proj", "lm_head", "output"];
    const QKV_SUBTYPES = ["q_proj", "k_proj", "v_proj"];

    type Edge = { src: string; tgt: string; val: number };

    type GraphArtifactData = {
        tokens: string[];
        edges: Edge[];
        outputProbs: Record<string, OutputProbability>;
        nodeCiVals: Record<string, number>;
        nodeSubcompActs: Record<string, number>;
        maxAbsAttr: number;
        l0_total: number;
    };

    type Props = {
        data: GraphArtifactData;
        caption?: string;
        componentGap?: number;
        layerGap?: number;
    };

    let { data, caption, componentGap = 4, layerGap = 24 }: Props = $props();

    // Refs
    let innerContainer: HTMLDivElement;

    // Zoom/pan
    const zoom = useZoomPan(() => innerContainer);

    type LayerInfo = {
        name: string;
        block: number;
        type: "attn" | "mlp" | "embed" | "output";
        subtype: string;
    };

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

    function getRowLabel(layer: string): string {
        const rowKey = getRowKey(layer);
        const isQkvGroup = rowKey.endsWith(".qkv");
        return getAliasedRowLabel(layer, isQkvGroup);
    }

    const maxAbsAttr = $derived(data.maxAbsAttr || 1);
    const maxCi = $derived.by(() => {
        let max = 0;
        for (const ci of Object.values(data.nodeCiVals)) {
            if (ci > max) max = ci;
        }
        return max || 1;
    });

    // All nodes from nodeCiVals
    const allNodes = $derived(new SvelteSet(Object.keys(data.nodeCiVals)));

    // Sort edges by magnitude, take top 500
    const filteredEdges = $derived.by(() => {
        const edgesCopy = [...data.edges];
        const sortedEdges = edgesCopy.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
        return sortedEdges.slice(0, 500);
    });

    // Build layout
    const { nodePositions, layerYPositions, seqXStarts, width, height } = $derived.by(() => {
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

        // Assign Y positions (output at top, wte at bottom)
        const rowYPositions: Record<string, number> = {};
        for (let i = 0; i < rows.length; i++) {
            const distanceFromEnd = rows.length - 1 - i;
            rowYPositions[rows[i]] = MARGIN.top + distanceFromEnd * (COMPONENT_SIZE + layerGap);
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
                        totalQkv += 2;
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

                // Sort by importance (CI)
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

        return {
            nodePositions,
            layerYPositions,
            seqXStarts,
            width: widthVal,
            height: heightVal,
        };
    });

    // SVG dimensions
    const svgWidth = $derived(width * zoom.scale + Math.max(zoom.translateX, 0));
    const svgHeight = $derived(height * zoom.scale + Math.max(zoom.translateY, 0));

    // Pre-compute node styles
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
                }
            } else {
                const ci = data.nodeCiVals[`${layer}:${seqIdx}:${cIdx}`] || 0;
                const intensity = ci / maxCi;
                opacity = 0.2 + intensity * 0.8;
            }

            styles[nodeKey] = { fill, opacity };
        }

        return styles;
    });

    // Build SVG edges string
    const edgesSvgString = $derived.by(() => {
        let svg = "";

        for (let i = filteredEdges.length - 1; i >= 0; i--) {
            const edge = filteredEdges[i];
            const p1 = nodePositions[edge.src];
            const p2 = nodePositions[edge.tgt];
            if (p1 && p2) {
                const color = getEdgeColor(edge.val);
                const w = lerp(1, 4, Math.abs(edge.val) / maxAbsAttr);
                const op = lerp(0.1, 0.6, Math.abs(edge.val) / maxAbsAttr);
                const dy = Math.abs(p2.y - p1.y);
                const curveOffset = Math.max(20, dy * 0.4);
                const cp1y = p1.y - curveOffset;
                const cp2y = p2.y + curveOffset;
                const d = `M ${p1.x},${p1.y} C ${p1.x},${cp1y} ${p2.x},${cp2y} ${p2.x},${p2.y}`;
                const title = `${edge.src} → ${edge.tgt}: ${edge.val.toFixed(4)}`;
                svg += `<path d="${d}" stroke="${color}" stroke-width="${w}" opacity="${op}" fill="none"><title>${title}</title></path>`;
            }
        }

        return svg;
    });

    function handlePanStart(event: MouseEvent) {
        const target = event.target as Element;
        if (target.closest(".node-group")) return;
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
        }
    }
</script>

<div class="artifact-graph">
    {#if caption}
        <div class="caption">{caption}</div>
    {/if}

    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="graph-wrapper"
        class:panning={zoom.isPanning}
        onmousedown={handlePanStart}
        onmousemove={zoom.updatePan}
        onmouseup={zoom.endPan}
        onmouseleave={zoom.endPan}
    >
        <ZoomControls scale={zoom.scale} onZoomIn={zoom.zoomIn} onZoomOut={zoom.zoomOut} onReset={zoom.reset} />

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
            <svg width={svgWidth} height={svgHeight}>
                <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
                    <!-- Edges -->
                    <g class="edges-layer">
                        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                        {@html edgesSvgString}
                    </g>

                    <!-- Nodes -->
                    <g class="nodes-layer">
                        {#each Object.entries(nodePositions) as [key, pos] (key)}
                            {@const style = nodeStyles[key]}
                            {@const [layer, seqIdx, cIdx] = key.split(":")}
                            {@const isOutput = layer === "output"}
                            {@const outputEntry = isOutput ? data.outputProbs[`${seqIdx}:${cIdx}`] : null}
                            {@const ci = data.nodeCiVals[key]}
                            {@const tooltip = isOutput
                                ? `"${outputEntry?.token ?? "?"}" (prob: ${outputEntry?.prob?.toFixed(2) ?? "?"})`
                                : `${layer}:${cIdx} @ pos ${seqIdx} (CI: ${ci?.toFixed(2) ?? "?"})`}
                            <g class="node-group">
                                <rect
                                    class="node"
                                    x={pos.x - COMPONENT_SIZE / 2}
                                    y={pos.y - COMPONENT_SIZE / 2}
                                    width={COMPONENT_SIZE}
                                    height={COMPONENT_SIZE}
                                    fill={style.fill}
                                    rx="1"
                                    opacity={style.opacity}
                                >
                                    <title>{tooltip}</title>
                                </rect>
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
                        {/each}
                    </g>
                </svg>
            </div>
        </div>
    </div>

    <div class="stats">
        L0: {data.l0_total} · Edges: {filteredEdges.length}
    </div>
</div>

<style>
    .artifact-graph {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        overflow: hidden;
        margin: var(--space-3) 0;
        background: var(--bg-surface);
    }

    .caption {
        padding: var(--space-2) var(--space-3);
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
        background: var(--bg-elevated);
        border-bottom: 1px solid var(--border-default);
    }

    .graph-wrapper {
        display: flex;
        overflow: hidden;
        position: relative;
        height: 400px;
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

    .node {
        transition: opacity var(--transition-normal);
    }

    .stats {
        padding: var(--space-1) var(--space-3);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        background: var(--bg-elevated);
        border-top: 1px solid var(--border-default);
    }
</style>
