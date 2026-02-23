<script lang="ts">
    import { getContext } from "svelte";
    import type { EdgeAttribution, OutputProbEntry, TokenInfo } from "../../lib/promptAttributionsTypes";
    import { formatNodeKeyForDisplay } from "../../lib/promptAttributionsTypes";
    import { RUN_KEY, type InterpretationBackendState, type RunContext } from "../../lib/useRun.svelte";
    import { lerp } from "../prompt-attr/graphUtils";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        items: EdgeAttribution[];
        onClick: (key: string) => void;
        pageSize: number;
        direction: "positive" | "negative";
        title?: string;
        // Optional: only needed for prompt-level attributions with wte/output pseudo-layers
        tokens?: string[];
        outputProbs?: Record<string, OutputProbEntry>;
    };

    let { items, onClick, pageSize, direction, title, tokens, outputProbs }: Props = $props();

    // Extract component key (layer:cIdx) from either format
    function getComponentKey(key: string): string {
        const parts = key.split(":");
        if (parts.length === 3) {
            return `${parts[0]}:${parts[2]}`; // layer:cIdx from layer:seq:cIdx
        }
        return key; // already layer:cIdx
    }

    function getInterpretation(key: string): InterpretationBackendState {
        const componentKey = getComponentKey(key);
        const interp = runState.getInterpretation(componentKey);
        if (interp.status === "loaded" && interp.data.status === "generated") return interp.data;
        return { status: "none" };
    }

    // Get display info for a key - returns label and whether it's a token (pseudo-layer) node
    // Token nodes (wte/output) show the token string; component nodes show interpretation label
    function getDisplayInfo(key: string): { label: string; isTokenNode: boolean; isOutputToken?: boolean } {
        const parts = key.split(":");

        // Handle prompt attributions with 3-part keys (layer:seq:cIdx)
        if (tokens && outputProbs && parts.length === 3) {
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
                    throw new Error(`EdgeAttributionList: output node ${key} not found in outputProbs`);
                }
                return { label: entry.token, isTokenNode: true };
            }
        }

        // Handle dataset attributions with 2-part keys (layer:cIdx)
        if (parts.length === 2) {
            const layer = parts[0];
            const cIdx = parts[1];

            // wte node in dataset attributions: single pseudo-component
            if (layer === "wte") {
                return { label: "Input Embeddings", isTokenNode: true };
            }

            // output nodes in dataset attributions: show token string
            // Format: output:tokenId where tokenId is the vocab index
            if (layer === "output") {
                const vocabIdx = parseInt(cIdx);
                // Tokens are guaranteed loaded when run is loaded (see useRun.svelte.ts)
                const tokens = (runState.allTokens as { status: "loaded"; data: TokenInfo[] }).data;
                const tokenInfo = tokens.find((t) => t.id === vocabIdx);
                if (!tokenInfo) throw new Error(`Token not found for vocab index ${vocabIdx}`);
                return { label: tokenInfo.string, isTokenNode: true, isOutputToken: true };
            }
        }

        // Component nodes: show interpretation label or "N/A"
        const interp = getInterpretation(key);

        if (interp.status === "generated")
            return {
                label: interp.data.label,
                isTokenNode: false,
            };

        if (interp.status === "generating") return { label: "Generating...", isTokenNode: false };

        return { label: "N/A", isTokenNode: false };
    }

    let currentPage = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

    // Track which pill is being hovered and its position
    let hoveredKey = $state<string | null>(null);
    let tooltipPosition = $state<{ top: number; left: number } | null>(null);

    function handleMouseEnter(key: string, event: MouseEvent) {
        hoveredKey = key;
        const target = event.currentTarget as HTMLElement;
        const rect = target.getBoundingClientRect();
        tooltipPosition = { top: rect.top, left: rect.left };
    }

    function handleMouseLeave() {
        hoveredKey = null;
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
    <div class="header-row">
        {#if title}
            <span class="list-title">{title}</span>
        {/if}
        {#if totalPages > 1}
            <div class="pagination">
                <button onclick={() => currentPage--} disabled={currentPage === 0}>&lt;</button>
                <span>{currentPage + 1} / {totalPages}</span>
                <button onclick={() => currentPage++} disabled={currentPage >= totalPages - 1}>&gt;</button>
            </div>
        {/if}
    </div>
    <div class="items">
        {#each paginatedItems as { key, value, normalizedMagnitude } (key)}
            {@const bgColor = getBgColor(normalizedMagnitude)}
            {@const textColor = normalizedMagnitude > 0.8 ? "white" : "var(--text-primary)"}
            {@const displayInfo = getDisplayInfo(key)}
            {@const interp = !displayInfo.isTokenNode ? getInterpretation(key) : undefined}
            {@const isHovered = hoveredKey === key}
            <div class="pill-container" onmouseenter={(e) => handleMouseEnter(key, e)} onmouseleave={handleMouseLeave}>
                <button class="edge-pill" style="background: {bgColor};" onclick={() => onClick(key)}>
                    <span class="node-key" style="color: {textColor};">{formatNodeKeyForDisplay(key)}</span>
                </button>
                {#if isHovered && tooltipPosition}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div
                        class="tooltip"
                        style="top: {tooltipPosition.top}px; left: {tooltipPosition.left}px;"
                        onmouseenter={() => (hoveredKey = key)}
                        onmouseleave={handleMouseLeave}
                    >
                        <div class="tooltip-value">Attribution: {value.toFixed(3)}</div>
                        {#if !displayInfo.isTokenNode && interp?.status === "generated"}
                            <button class="tooltip-label copyable" onclick={() => copyToClipboard(interp.data.label)}>
                                {interp.data.label}
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
                            <div class="tooltip-confidence">Confidence: {interp.data.confidence}</div>
                        {:else if displayInfo.isTokenNode}
                            <div class="tooltip-token">Token: {displayInfo.label}</div>
                        {/if}
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

    .header-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        min-height: 1.25rem; /* Ensure consistent height even when empty */
    }

    .list-title {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-style: italic;
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

    .node-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
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

    .tooltip-value {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--space-1);
    }

    .tooltip-token {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-secondary);
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

    .tooltip-confidence {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
