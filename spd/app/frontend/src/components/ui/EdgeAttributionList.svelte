<script lang="ts">
    import { getContext } from "svelte";
    import type { EdgeAttribution, OutputProbEntry } from "../../lib/localAttributionsTypes";
    import { formatNodeKeyForDisplay } from "../../lib/localAttributionsTypes";
    import type { Interpretation } from "../../lib/api";
    import { RUN_STATE_KEY, type RunStateContext } from "../../lib/runState.svelte";
    import { lerp } from "../local-attr/graphUtils";

    const runState = getContext<RunStateContext>(RUN_STATE_KEY);

    type Props = {
        items: EdgeAttribution[];
        onNodeClick: (nodeKey: string) => void;
        pageSize: number;
        direction: "positive" | "negative";
        tokens: string[];
        outputProbs: Record<string, OutputProbEntry>;
    };

    let { items, onNodeClick, pageSize, direction, tokens, outputProbs }: Props = $props();

    // Extract component key (layer:cIdx) from node key (layer:seq:cIdx)
    function getComponentKey(nodeKey: string): string {
        const parts = nodeKey.split(":");
        return `${parts[0]}:${parts[2]}`; // layer:cIdx
    }

    function getInterpretation(nodeKey: string): Interpretation | undefined {
        const componentKey = getComponentKey(nodeKey);
        return runState.getInterpretation(componentKey);
    }

    // Get display info for a node - returns label and whether it's a token (pseudo-layer) node
    // Token nodes (wte/output) show the token string; component nodes show interpretation label
    function getNodeDisplayInfo(nodeKey: string): { label: string; isTokenNode: boolean } {
        const parts = nodeKey.split(":");
        const layer = parts[0];
        const seqIdx = parseInt(parts[1]);
        const cIdx = parts[2];

        // wte (input embedding) nodes: show the token at this sequence position
        if (layer === "wte") {
            if (seqIdx < 0 || seqIdx >= tokens.length) {
                throw new Error(
                    `EdgeAttributionList: seqIdx ${seqIdx} out of bounds for tokens length ${tokens.length}`,
                );
            }
            return { label: tokens[seqIdx], isTokenNode: true };
        }

        // output nodes: show the predicted token string
        if (layer === "output") {
            const entry = outputProbs[`${seqIdx}:${cIdx}`];
            if (!entry) {
                throw new Error(`EdgeAttributionList: output node ${nodeKey} not found in outputProbs`);
            }
            return { label: entry.token, isTokenNode: true };
        }

        // Component nodes: show interpretation label or "N/A"
        const interp = getInterpretation(nodeKey);
        return { label: interp?.label ?? "N/A", isTokenNode: false };
    }

    let currentPage = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

    // Track which pill is being hovered and its position
    let hoveredNodeKey = $state<string | null>(null);
    let tooltipPosition = $state<{ top: number; left: number } | null>(null);

    function handleMouseEnter(nodeKey: string, event: MouseEvent) {
        hoveredNodeKey = nodeKey;
        const target = event.currentTarget as HTMLElement;
        const rect = target.getBoundingClientRect();
        tooltipPosition = { top: rect.top, left: rect.left };
    }

    function handleMouseLeave() {
        hoveredNodeKey = null;
        tooltipPosition = null;
    }

    // Reset page when items change
    $effect(() => {
        items; // eslint-disable-line @typescript-eslint/no-unused-expressions
        currentPage = 0;
    });

    function getBgColor(normalizedMagnitude: number): string {
        const intensity = lerp(0, 0.8, normalizedMagnitude);
        if (direction === "negative") {
            return `rgba(220, 38, 38, ${intensity})`; // red
        }
        return `rgba(22, 74, 193, ${intensity})`; // blue
    }

    async function copyToClipboard(text: string) {
        await navigator.clipboard.writeText(text);
    }
</script>

<div class="edge-attribution-list">
    {#if totalPages > 1}
        <div class="pagination">
            <button onclick={() => currentPage--} disabled={currentPage === 0}>&lt;</button>
            <span>{currentPage + 1} / {totalPages}</span>
            <button onclick={() => currentPage++} disabled={currentPage >= totalPages - 1}>&gt;</button>
        </div>
    {/if}
    <div class="items">
        {#each paginatedItems as { nodeKey, value, normalizedMagnitude } (nodeKey)}
            {@const bgColor = getBgColor(normalizedMagnitude)}
            {@const textColor = normalizedMagnitude > 0.8 ? "white" : "var(--text-primary)"}
            {@const displayInfo = getNodeDisplayInfo(nodeKey)}
            {@const interp = !displayInfo.isTokenNode ? getInterpretation(nodeKey) : undefined}
            {@const isHovered = hoveredNodeKey === nodeKey}
            <div
                class="pill-container"
                onmouseenter={(e) => handleMouseEnter(nodeKey, e)}
                onmouseleave={handleMouseLeave}
            >
                <button class="edge-pill" style="background: {bgColor};" onclick={() => onNodeClick(nodeKey)}>
                    <span class="interp-label" style="color: {textColor};">{displayInfo.label}</span>
                    <span class="value" style="color: {textColor};">{value.toFixed(2)}</span>
                </button>
                {#if isHovered && !displayInfo.isTokenNode && interp && tooltipPosition}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div
                        class="tooltip"
                        style="top: {tooltipPosition.top}px; left: {tooltipPosition.left}px;"
                        onmouseenter={() => (hoveredNodeKey = nodeKey)}
                        onmouseleave={handleMouseLeave}
                    >
                        <div class="tooltip-key">{formatNodeKeyForDisplay(nodeKey)}</div>
                        <button class="tooltip-label copyable" onclick={() => copyToClipboard(interp.label)}>
                            {interp.label}
                            <svg
                                class="copy-icon"
                                width="12"
                                height="12"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                            >
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        </button>
                        <div class="tooltip-reasoning">{interp.reasoning}</div>
                        <div class="tooltip-confidence">Confidence: {interp.confidence}</div>
                    </div>
                {/if}
            </div>
        {/each}
    </div>
</div>

<style>
    .edge-attribution-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .items {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        background: var(--bg-elevated);
        padding: var(--space-2);
        border: 1px solid var(--border-default);
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .pagination button {
        padding: 2px 6px;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-xs);
    }

    .pagination button:hover:not(:disabled) {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .pagination button:disabled {
        opacity: 0.4;
        cursor: default;
    }

    .pill-container {
        position: relative;
    }

    .edge-pill {
        display: inline-flex;
        align-items: center;
        gap: var(--space-2);
        padding: 2px 4px;
        border-radius: 3px;
        white-space: nowrap;
        cursor: default;
        border: 1px solid var(--border-default);
        font-family: inherit;
        font-size: inherit;
    }

    .value {
        opacity: 0.8;
    }

    .interp-label {
        font-family: var(--font-sans);
        font-weight: 500;
        max-width: 150px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .tooltip {
        position: fixed;
        transform: translateY(-100%);
        margin-top: -8px;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        z-index: 10000;
        min-width: 200px;
        max-width: 350px;
    }

    .tooltip-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
        margin-bottom: var(--space-1);
    }

    .tooltip-label {
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: var(--text-sm);
        color: var(--text-primary);
        margin-bottom: var(--space-1);
    }

    .tooltip-label.copyable {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        background: none;
        border: none;
        padding: 2px 4px;
        margin: -2px -4px;
        margin-bottom: var(--space-1);
        border-radius: 3px;
        cursor: pointer;
        text-align: left;
    }

    .tooltip-label.copyable:hover {
        background: var(--bg-surface);
    }

    .tooltip-label.copyable .copy-icon {
        opacity: 0.4;
        flex-shrink: 0;
    }

    .tooltip-label.copyable:hover .copy-icon {
        opacity: 0.8;
    }

    .tooltip-reasoning {
        font-family: var(--font-sans);
        font-size: var(--text-xs);
        color: var(--text-secondary);
        line-height: 1.4;
        margin-bottom: var(--space-1);
    }

    .tooltip-confidence {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
