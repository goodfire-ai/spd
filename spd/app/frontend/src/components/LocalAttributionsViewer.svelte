<script lang="ts">
    import type {
        LocalAttributionsData,
        LayerInfo,
        EdgeData,
        NodePosition,
        PinnedNode,
    } from "./localAttributionsTypes";

    // Props - pass data in from parent component
    type Props = {
        data: LocalAttributionsData;
    };

    let { data }: Props = $props();

    // Constants
    const COMPONENT_SIZE = 8;
    const COMPONENT_GAP = 2;
    const LAYER_GAP = 30;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;

    const SUBTYPE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"];
    const COLORS: Record<string, string> = {
        q_proj: "#e91e63",
        k_proj: "#9c27b0",
        v_proj: "#673ab7",
        o_proj: "#3f51b5",
        c_fc: "#009688",
        down_proj: "#4caf50",
    };

    // State
    let maxMeanCI = $state(0.5);
    let topK = $state(100);
    let hoveredNodeKey = $state<string | null>(null);
    let isHoveringTooltip = $state(false);
    let pinnedNodes = $state<PinnedNode[]>([]);
    let tooltipX = $state(0);
    let tooltipY = $state(0);
    let tooltipLayer = $state("");
    let tooltipCIdx = $state(0);

    // Parse layer name
    function parseLayer(name: string): LayerInfo | null {
        const m = name.match(/h\.(\d+)\.(attn|mlp)\.(\w+)/);
        if (!m) return null;
        return {
            name,
            block: +m[1],
            type: m[2] as "attn" | "mlp",
            subtype: m[3],
        };
    }

    // Compute component importance and edges
    const { allEdges, componentImportance, componentImportanceLocal, maxAttr } = $derived.by(() => {
        const componentImportance: Record<string, number> = {};
        const componentImportanceLocal: Record<string, number> = {};
        const allEdges: EdgeData[] = [];
        let maxAttr = 1;

        for (const pair of data.pairs) {
            const isCrossSeq = pair.is_cross_seq;
            const cInIdxs = pair.trimmed_c_in_idxs;
            const cOutIdxs = pair.trimmed_c_out_idxs;

            for (const entry of pair.attribution) {
                let sIn: number, cInLocal: number, sOut: number, cOutLocal: number, val: number;

                if (isCrossSeq) {
                    [sIn, cInLocal, sOut, cOutLocal, val] = entry as [number, number, number, number, number];
                } else {
                    [sIn, cInLocal, cOutLocal, val] = entry as [number, number, number, number];
                    sOut = sIn;
                }

                const cInIdx = cInIdxs[cInLocal];
                const cOutIdx = cOutIdxs[cOutLocal];

                allEdges.push({
                    srcKey: `${pair.source}:${sIn}:${cInIdx}`,
                    tgtKey: `${pair.target}:${sOut}:${cOutIdx}`,
                    val,
                });

                const absVal = Math.abs(val);
                if (absVal > maxAttr) maxAttr = absVal;

                const valSq = val * val;

                const srcKey = `${pair.source}:${cInIdx}`;
                componentImportance[srcKey] = (componentImportance[srcKey] || 0) + valSq;

                const tgtKey = `${pair.target}:${cOutIdx}`;
                componentImportance[tgtKey] = (componentImportance[tgtKey] || 0) + valSq;

                const srcKeyLocal = `${pair.source}:${sIn}:${cInIdx}`;
                componentImportanceLocal[srcKeyLocal] = (componentImportanceLocal[srcKeyLocal] || 0) + valSq;

                const tgtKeyLocal = `${pair.target}:${sOut}:${cOutIdx}`;
                componentImportanceLocal[tgtKeyLocal] = (componentImportanceLocal[tgtKeyLocal] || 0) + valSq;
            }
        }

        allEdges.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));

        return { allEdges, componentImportance, componentImportanceLocal, maxAttr };
    });

    // Build mean CI lookup
    const componentMeanCI = $derived.by(() => {
        const lookup: Record<string, number> = {};
        if (data.activation_contexts) {
            for (const [layer, subcomps] of Object.entries(data.activation_contexts)) {
                for (const subcomp of subcomps) {
                    lookup[`${layer}:${subcomp.subcomponent_idx}`] = subcomp.mean_ci;
                }
            }
        }
        return lookup;
    });

    // Filter edges by mean CI and topK
    const { filteredEdges, activeNodes } = $derived.by(() => {
        const ciFilteredEdges = allEdges.filter((edge) => {
            const [srcLayer, , srcCIdx] = edge.srcKey.split(":");
            const [tgtLayer, , tgtCIdx] = edge.tgtKey.split(":");
            const srcMeanCI = componentMeanCI[`${srcLayer}:${srcCIdx}`];
            const tgtMeanCI = componentMeanCI[`${tgtLayer}:${tgtCIdx}`];
            const srcOk = srcMeanCI === undefined || srcMeanCI <= maxMeanCI;
            const tgtOk = tgtMeanCI === undefined || tgtMeanCI <= maxMeanCI;
            return srcOk && tgtOk;
        });

        const topEdges = ciFilteredEdges.slice(0, topK);

        const activeNodes = new Set<string>();
        for (const edge of topEdges) {
            activeNodes.add(edge.srcKey);
            activeNodes.add(edge.tgtKey);
        }

        return { filteredEdges: topEdges, activeNodes };
    });

    // Build layout
    const { nodePositions, layerYPositions, seqWidths, seqXStarts, width, height } = $derived.by(() => {
        const nodesPerLayerSeq: Record<string, number[]> = {};
        const allLayers = new Set<string>();

        for (const nodeKey of activeNodes) {
            const [layer, seqIdx, cIdx] = nodeKey.split(":");
            allLayers.add(layer);
            const key = `${layer}:${seqIdx}`;
            if (!nodesPerLayerSeq[key]) nodesPerLayerSeq[key] = [];
            nodesPerLayerSeq[key].push(+cIdx);
        }

        const layers = Array.from(allLayers)
            .map(parseLayer)
            .filter((l): l is LayerInfo => l !== null)
            .sort((a, b) => {
                if (a.block !== b.block) return a.block - b.block;
                return SUBTYPE_ORDER.indexOf(a.subtype) - SUBTYPE_ORDER.indexOf(b.subtype);
            });

        const layerYPositions: Record<string, number> = {};
        let currentY = MARGIN.top;
        for (const layer of layers.slice().reverse()) {
            layerYPositions[layer.name] = currentY;
            currentY += COMPONENT_SIZE + LAYER_GAP;
        }

        const maxComponentsPerSeq: number[] = [];
        for (let seqIdx = 0; seqIdx < data.tokens.length; seqIdx++) {
            let maxAtSeq = 0;
            for (const layer of layers) {
                const key = `${layer.name}:${seqIdx}`;
                const count = nodesPerLayerSeq[key]?.length || 0;
                maxAtSeq = Math.max(maxAtSeq, count);
            }
            maxComponentsPerSeq.push(maxAtSeq);
        }

        const MIN_COL_WIDTH = 30;
        const COL_PADDING = 16;
        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + COMPONENT_GAP) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        const nodePositions: Record<string, NodePosition> = {};

        for (const layer of layers) {
            for (let seqIdx = 0; seqIdx < data.tokens.length; seqIdx++) {
                const key = `${layer.name}:${seqIdx}`;
                const components = nodesPerLayerSeq[key] || [];

                components.sort((a, b) => {
                    const impA = componentImportanceLocal[`${layer.name}:${seqIdx}:${a}`] || 0;
                    const impB = componentImportanceLocal[`${layer.name}:${seqIdx}:${b}`] || 0;
                    return impB - impA;
                });

                const baseX = seqXStarts[seqIdx] + COL_PADDING;
                const baseY = layerYPositions[layer.name];

                for (let i = 0; i < components.length; i++) {
                    const cIdx = components[i];
                    const nodeKey = `${layer.name}:${seqIdx}:${cIdx}`;
                    nodePositions[nodeKey] = {
                        x: baseX + i * (COMPONENT_SIZE + COMPONENT_GAP) + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                }
            }
        }

        const totalSeqWidth = seqXStarts[seqXStarts.length - 1] + seqWidths[seqWidths.length - 1];
        const width = totalSeqWidth + MARGIN.right;

        let maxY = 0;
        for (const layer in layerYPositions) {
            maxY = Math.max(maxY, layerYPositions[layer] + COMPONENT_SIZE);
        }
        const height = maxY + MARGIN.bottom;

        return { nodePositions, layerYPositions, seqWidths, seqXStarts, width, height };
    });

    // Compute node output attributions
    const { nodeOutAttr, maxNodeOutAttr } = $derived.by(() => {
        const nodeOutAttr: Record<string, number> = {};
        for (const pair of data.pairs) {
            const isCrossSeq = pair.is_cross_seq;
            const cInIdxs = pair.trimmed_c_in_idxs;

            for (const entry of pair.attribution) {
                let sIn: number, cInLocal: number, val: number;
                if (isCrossSeq) {
                    [sIn, cInLocal, , , val] = entry as [number, number, number, number, number];
                } else {
                    [sIn, cInLocal, , val] = entry as [number, number, number, number];
                }

                const cIdx = cInIdxs[cInLocal];
                const key = `${pair.source}:${sIn}:${cIdx}`;
                nodeOutAttr[key] = (nodeOutAttr[key] || 0) + Math.abs(val);
            }
        }

        const maxNodeOutAttr = Math.max(...Object.values(nodeOutAttr), 1);
        return { nodeOutAttr, maxNodeOutAttr };
    });

    // Get highlighted node keys for same component
    const highlightedNodeKeys = $derived.by(() => {
        if (pinnedNodes.length === 0) return new Set<string>();

        const keys = new Set<string>();
        for (const pinned of pinnedNodes) {
            for (const nodeKey in nodePositions) {
                const [layer, _seqIdx, cIdx] = nodeKey.split(":");
                if (layer === pinned.layer && +cIdx === pinned.cIdx) {
                    keys.add(nodeKey);
                }
            }
        }
        return keys;
    });

    // Highlight logic for hover
    const hoverHighlightedKeys = $derived.by(() => {
        if (!hoveredNodeKey) return new Set<string>();

        const [layer, , cIdx] = hoveredNodeKey.split(":");
        const keys = new Set<string>();
        for (const nodeKey in nodePositions) {
            const [nLayer, , nCIdx] = nodeKey.split(":");
            if (nLayer === layer && nCIdx === cIdx) {
                keys.add(nodeKey);
            }
        }
        return keys;
    });

    // Combined highlight keys (hover or pinned)
    const allHighlightedKeys = $derived(pinnedNodes.length > 0 ? highlightedNodeKeys : hoverHighlightedKeys);

    // Event handlers
    function handleNodeMouseEnter(e: MouseEvent, layer: string, _seqIdx: string, cIdx: number, nodeKey: string) {
        hoveredNodeKey = nodeKey;
        tooltipX = e.clientX;
        tooltipY = e.clientY;
        tooltipLayer = layer;
        tooltipCIdx = cIdx;
    }

    function handleNodeMouseLeave() {
        hoveredNodeKey = null;
        setTimeout(() => {
            if (!isHoveringTooltip && !hoveredNodeKey) {
                tooltipLayer = "";
            }
        }, 100);
    }

    function handleNodeClick(layer: string, cIdx: number, nodeKey: string) {
        const existingIdx = pinnedNodes.findIndex((p) => p.layer === layer && p.cIdx === cIdx);

        if (existingIdx >= 0) {
            pinnedNodes = pinnedNodes.filter((_, i) => i !== existingIdx);
        } else {
            pinnedNodes = [...pinnedNodes, { layer, cIdx, nodeKey }];
        }

        tooltipLayer = "";
        hoveredNodeKey = null;
    }

    function unpinNode(layer: string, cIdx: number) {
        pinnedNodes = pinnedNodes.filter((p) => !(p.layer === layer && p.cIdx === cIdx));
    }

    // Build full node details HTML
    function buildNodeDetailsData(layer: string, cIdx: number) {
        const globalImp = componentImportance[`${layer}:${cIdx}`] || 0;
        let sumAbsAttr = 0;
        let countEdges = 0;
        for (const edge of allEdges) {
            const [srcLayer, , srcCIdx] = edge.srcKey.split(":");
            const [tgtLayer, , tgtCIdx] = edge.tgtKey.split(":");
            if ((srcLayer === layer && +srcCIdx === cIdx) || (tgtLayer === layer && +tgtCIdx === cIdx)) {
                sumAbsAttr += Math.abs(edge.val);
                countEdges++;
            }
        }
        const meanAbsAttr = countEdges > 0 ? sumAbsAttr / countEdges : 0;
        const meanSqAttr = countEdges > 0 ? globalImp / countEdges : 0;

        const actCtx = data.activation_contexts?.[layer];
        const subcomp = actCtx?.find((s) => s.subcomponent_idx === cIdx);

        return {
            layer,
            cIdx,
            globalImp,
            meanAbsAttr,
            meanSqAttr,
            countEdges,
            subcomp,
        };
    }

    // Get tooltip details
    const tooltipDetails = $derived(
        tooltipLayer && hoveredNodeKey ? buildNodeDetailsData(tooltipLayer, tooltipCIdx) : null,
    );

    // Tooltip position
    const tooltipStyle = $derived.by(() => {
        if (!tooltipDetails) return "";
        const padding = 15;
        let left = tooltipX + padding;
        let top = tooltipY + padding;

        if (typeof window !== "undefined") {
            if (left + 1000 > window.innerWidth) {
                left = tooltipX - 1000 - padding;
            }
            if (top + 800 > window.innerHeight) {
                top = tooltipY - 800 - padding;
            }
        }

        return `left: ${left}px; top: ${top}px;`;
    });
</script>

<div class="local-attributions-viewer">
    <h1>Local Attributions Graph</h1>

    <div class="controls">
        <label>
            Max mean CI:
            <input type="number" min="0" max="1" step="0.01" bind:value={maxMeanCI} />
        </label>
        <label>
            Top K edges:
            <input type="number" min="10" max="10000" step="100" bind:value={topK} />
        </label>
        <span class="edge-count">Showing {filteredEdges.length} edges</span>
    </div>

    <div class="graph-wrapper">
        <div class="layer-labels-container" style="width: {LABEL_WIDTH}px;">
            <svg width={LABEL_WIDTH} {height} style="display: block;">
                {#each Object.entries(layerYPositions) as [layer, y]}
                    {@const info = parseLayer(layer)}
                    {#if info}
                        <text
                            x={LABEL_WIDTH - 10}
                            y={y + COMPONENT_SIZE / 2}
                            text-anchor="end"
                            dominant-baseline="middle"
                            font-size="11"
                            font-weight="500"
                            fill={COLORS[info.subtype]}
                        >
                            {info.subtype}
                        </text>
                    {/if}
                {/each}
            </svg>
        </div>

        <div class="graph-container">
            <svg {width} {height}>
                <!-- Edges -->
                {#each filteredEdges as edge}
                    {@const p1 = nodePositions[edge.srcKey]}
                    {@const p2 = nodePositions[edge.tgtKey]}
                    {#if p1 && p2}
                        {@const color = edge.val > 0 ? "#2196f3" : "#f44336"}
                        {@const w = Math.max(0.5, (Math.abs(edge.val) / maxAttr) * 2)}
                        {@const op = 0.05 + (Math.abs(edge.val) / maxAttr) * 0.95}
                        {@const isHighlighted =
                            allHighlightedKeys.has(edge.srcKey) || allHighlightedKeys.has(edge.tgtKey)}
                        <line
                            class="edge"
                            class:highlighted={isHighlighted}
                            x1={p1.x}
                            y1={p1.y}
                            x2={p2.x}
                            y2={p2.y}
                            stroke={color}
                            stroke-width={w}
                            opacity={op}
                        />
                    {/if}
                {/each}

                <!-- Nodes -->
                {#each Object.entries(nodePositions) as [key, pos]}
                    {@const [layer, seqIdx, cIdx] = key.split(":")}
                    {@const info = parseLayer(layer)}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    {#if info}
                        {@const outAttr = nodeOutAttr[key] || 0}
                        {@const intensity = Math.min(1, outAttr / maxNodeOutAttr)}
                        {@const baseColor = COLORS[info.subtype] || "#999"}
                        {@const opacity = 0.2 + intensity * 0.8}
                        {@const isSameComponent = allHighlightedKeys.has(key)}
                        <!-- svelte-ignore a11y_click_events_have_key_events -->
                        <rect
                            class="node"
                            class:same-component={isSameComponent}
                            data-layer={layer}
                            data-seq={seqIdx}
                            data-cidx={cIdx}
                            x={pos.x - COMPONENT_SIZE / 2}
                            y={pos.y - COMPONENT_SIZE / 2}
                            width={COMPONENT_SIZE}
                            height={COMPONENT_SIZE}
                            fill={baseColor}
                            rx="1"
                            {opacity}
                            onmouseenter={(e) => handleNodeMouseEnter(e, layer, seqIdx, +cIdx, key)}
                            onmouseleave={handleNodeMouseLeave}
                            onclick={() => handleNodeClick(layer, +cIdx, key)}
                        >
                            <title>{layer} seq={seqIdx} c={cIdx}\nOut attr: {outAttr.toFixed(4)}</title>
                        </rect>
                    {/if}
                {/each}
            </svg>

            <div class="token-labels-container">
                <svg {width} height="50" style="display: block;">
                    {#each data.tokens as token, i}
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
                        <text x={colCenter} y="38" text-anchor="middle" font-size="10" fill="#999">
                            [{i}]
                        </text>
                    {/each}
                </svg>
            </div>
        </div>
    </div>

    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-items">
            {#each Object.entries(COLORS) as [name, color]}
                <div class="legend-item">
                    <span class="legend-dot" style="background: {color}"></span>
                    {name}
                </div>
            {/each}
        </div>
        <div class="legend-edge">
            <span class="edge-pos"></span> Positive
            <span class="edge-neg"></span> Negative
        </div>
    </div>

    {#if pinnedNodes.length > 0}
        <div class="pinned-container">
            <h3>Pinned Components</h3>
            <div class="pinned-items">
                {#each pinnedNodes as pinned}
                    {@const details = buildNodeDetailsData(pinned.layer, pinned.cIdx)}
                    <div class="pinned-detail">
                        <div class="pinned-header">
                            <strong>{details.layer} : {details.cIdx}</strong>
                            <button onclick={() => unpinNode(pinned.layer, pinned.cIdx)}>✕ unpin</button>
                        </div>

                        {#if details.subcomp}
                            <p class="stats">
                                <strong>Mean CI:</strong>
                                {details.subcomp.mean_ci.toFixed(4)} |
                                <strong>Sum attr²:</strong>
                                {details.globalImp.toFixed(6)} |
                                <strong>Mean |attr|:</strong>
                                {details.meanAbsAttr.toFixed(6)} |
                                <strong>Mean attr²:</strong>
                                {details.meanSqAttr.toFixed(6)}
                                <small>({details.countEdges} edges)</small>
                            </p>

                            {#if details.subcomp.example_tokens && details.subcomp.example_tokens.length > 0}
                                <h4>Top Activating Examples</h4>
                                {#each details.subcomp.example_tokens as tokens, i}
                                    {@const ciVals = details.subcomp.example_ci[i]}
                                    {@const activePos = details.subcomp.example_active_pos[i]}
                                    {@const activeCi = details.subcomp.example_active_ci[i]}
                                    <div class="example-row">
                                        {#each tokens as token, j}
                                            {@const ci = ciVals[j]}
                                            {@const isActive = j === activePos}
                                            {@const bg = `rgba(255, 100, 100, ${Math.min(1, ci * 5)})`}
                                            <span
                                                class="example-token"
                                                class:active={isActive}
                                                style="background: {bg}"
                                                title="CI: {ci.toFixed(3)}"
                                            >
                                                {token}
                                            </span>
                                        {/each}
                                        <small>(CI: {activeCi.toFixed(3)})</small>
                                    </div>
                                {/each}
                            {/if}

                            <div class="token-tables">
                                {#if details.subcomp.pr_tokens && details.subcomp.pr_tokens.length > 0}
                                    <div>
                                        <h4>Top Input Tokens</h4>
                                        <table class="pr-table">
                                            <thead>
                                                <tr><th>Token</th><th title="P(fires | token)">P(fire|tok)</th></tr>
                                            </thead>
                                            <tbody>
                                                {#each details.subcomp.pr_tokens.slice(0, 10) as token, i}
                                                    <tr>
                                                        <td><code>{token}</code></td>
                                                        <td>{details.subcomp.pr_precisions?.[i].toFixed(3)}</td>
                                                    </tr>
                                                {/each}
                                            </tbody>
                                        </table>
                                    </div>
                                {/if}

                                {#if details.subcomp.predicted_tokens && details.subcomp.predicted_tokens.length > 0}
                                    <div>
                                        <h4>Top Predicted Tokens</h4>
                                        <table class="pr-table">
                                            <thead>
                                                <tr><th>Token</th><th>P(pred|fire)</th></tr>
                                            </thead>
                                            <tbody>
                                                {#each details.subcomp.predicted_tokens.slice(0, 10) as token, i}
                                                    <tr>
                                                        <td><code>{token}</code></td>
                                                        <td>{details.subcomp.predicted_probs?.[i].toFixed(3)}</td>
                                                    </tr>
                                                {/each}
                                            </tbody>
                                        </table>
                                    </div>
                                {/if}
                            </div>
                        {:else}
                            <p class="no-data">No activation context data for this component</p>
                        {/if}
                    </div>
                {/each}
            </div>
        </div>
    {/if}

    {#if tooltipDetails}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
            class="node-details"
            style={tooltipStyle}
            onmouseenter={() => (isHoveringTooltip = true)}
            onmouseleave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
        >
            <h3>Component Details</h3>
            <div class="node-info">
                <p><strong>Layer:</strong> {tooltipDetails.layer}</p>
                <p><strong>Component:</strong> {tooltipDetails.cIdx}</p>
                {#if tooltipDetails.subcomp}
                    <p><strong>Mean CI:</strong> {tooltipDetails.subcomp.mean_ci.toFixed(4)}</p>
                    <p><strong>Sum attr²:</strong> {tooltipDetails.globalImp.toFixed(6)}</p>
                    <p>
                        <strong>Mean |attr|:</strong>
                        {tooltipDetails.meanAbsAttr.toFixed(6)}
                        <small>({tooltipDetails.countEdges} edges)</small>
                    </p>
                    <p><strong>Mean attr²:</strong> {tooltipDetails.meanSqAttr.toFixed(6)}</p>
                {/if}
            </div>

            {#if tooltipDetails.subcomp}
                <div class="node-examples">
                    <h4>Top Activating Examples</h4>
                    {#each tooltipDetails.subcomp.example_tokens as tokens, i}
                        {@const ciVals = tooltipDetails.subcomp.example_ci[i]}
                        {@const activePos = tooltipDetails.subcomp.example_active_pos[i]}
                        {@const activeCi = tooltipDetails.subcomp.example_active_ci[i]}
                        <div class="example-row">
                            {#each tokens as token, j}
                                {@const ci = ciVals[j]}
                                {@const isActive = j === activePos}
                                {@const bg = `rgba(255, 100, 100, ${Math.min(1, ci * 5)})`}
                                <span
                                    class="example-token"
                                    class:active={isActive}
                                    style="background: {bg}"
                                    title="CI: {ci.toFixed(3)}"
                                >
                                    {token}
                                </span>
                            {/each}
                            <small>(CI: {activeCi.toFixed(3)})</small>
                        </div>
                    {/each}

                    {#if tooltipDetails.subcomp.pr_tokens && tooltipDetails.subcomp.pr_tokens.length > 0}
                        <h4>Top Input Tokens</h4>
                        <table class="pr-table">
                            <thead>
                                <tr><th>Token</th><th title="P(fires | token)">P(fire|tok)</th></tr>
                            </thead>
                            <tbody>
                                {#each tooltipDetails.subcomp.pr_tokens.slice(0, 10) as token, i}
                                    <tr>
                                        <td><code>{token}</code></td>
                                        <td>{tooltipDetails.subcomp.pr_precisions?.[i].toFixed(3)}</td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    {/if}

                    {#if tooltipDetails.subcomp.predicted_tokens && tooltipDetails.subcomp.predicted_tokens.length > 0}
                        <h4>Top Predicted Tokens</h4>
                        <table class="pr-table">
                            <thead>
                                <tr><th>Token</th><th>P(pred|fire)</th></tr>
                            </thead>
                            <tbody>
                                {#each tooltipDetails.subcomp.predicted_tokens.slice(0, 10) as token, i}
                                    <tr>
                                        <td><code>{token}</code></td>
                                        <td>{tooltipDetails.subcomp.predicted_probs?.[i].toFixed(3)}</td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    {/if}
                </div>
            {/if}
        </div>
    {/if}
</div>

<style>
    * {
        box-sizing: border-box;
    }

    .local-attributions-viewer {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        padding: 1rem;
        background: #fafafa;
    }

    h1 {
        margin-top: 0;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        flex-wrap: wrap;
    }

    .controls label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
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

    .graph-wrapper {
        display: flex;
        max-height: 80vh;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
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

    .legend {
        margin-top: 1rem;
        padding: 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
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

    .edge {
        transition:
            opacity 0.15s,
            stroke-width 0.15s;
    }

    .edge.highlighted {
        opacity: 1 !important;
        stroke-width: 3 !important;
    }

    .node {
        transition:
            stroke-width 0.15s,
            filter 0.15s;
        cursor: pointer;
    }

    .node.same-component {
        stroke: #000 !important;
        stroke-width: 2.5 !important;
        filter: brightness(0.7) saturate(1.5);
        opacity: 1 !important;
    }

    .node-details {
        position: fixed;
        padding: 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        max-width: 1000px;
        max-height: 800px;
        overflow-y: auto;
        z-index: 1000;
        pointer-events: auto;
    }

    .node-details h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }

    .node-info p {
        margin: 0.15rem 0;
        font-size: 0.85rem;
    }

    .node-examples h4 {
        margin: 0.75rem 0 0.25rem 0;
        font-size: 0.85rem;
    }

    .example-row {
        display: flex;
        flex-wrap: wrap;
        gap: 2px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }

    .example-token {
        padding: 2px 4px;
        border-radius: 2px;
    }

    .example-token.active {
        font-weight: bold;
        border: 1px solid #333;
    }

    .pr-table {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .pr-table th,
    .pr-table td {
        padding: 0.25rem 0.5rem;
        text-align: left;
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
    }

    .pinned-items {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }

    .pinned-detail {
        flex-shrink: 0;
        min-width: 350px;
        max-width: 500px;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        background: #fafafa;
    }

    .pinned-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .pinned-header strong {
        font-size: 1rem;
    }

    .pinned-header button {
        cursor: pointer;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
    }

    .stats {
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }

    .pinned-detail h4 {
        margin: 0.75rem 0 0.25rem 0;
        font-size: 0.85rem;
    }

    .token-tables {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }

    .no-data {
        color: #666;
        font-size: 0.85rem;
    }
</style>
