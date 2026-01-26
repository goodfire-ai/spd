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

    // Check if a key refers to a pseudo-layer token node (wte/output)
    function isTokenNode(key: string): boolean {
        const layer = key.split(":")[0];
        return layer === "wte" || layer === "output";
    }

    // Get display label for token nodes (wte/output pseudo-layers)
    function getTokenLabel(key: string): string {
        const parts = key.split(":");

        // Prompt attributions: 3-part keys (layer:seq:cIdx)
        if (tokens && outputProbs && parts.length === 3) {
            const [layer, seqStr, cIdx] = parts;
            const seqIdx = parseInt(seqStr);

            if (layer === "wte") {
                if (seqIdx < 0 || seqIdx >= tokens.length) {
                    throw new Error(
                        `EdgeAttributionList: seqIdx ${seqIdx} out of bounds for tokens length ${tokens.length}`,
                    );
                }
                return tokens[seqIdx];
            }

            if (layer === "output") {
                const entry = outputProbs[`${seqIdx}:${cIdx}`];
                if (!entry) {
                    throw new Error(`EdgeAttributionList: output node ${key} not found in outputProbs`);
                }
                return entry.token;
            }
        }

        // Dataset attributions: 2-part keys (layer:cIdx)
        if (parts.length === 2) {
            const [layer, cIdx] = parts;

            if (layer === "wte") {
                return "Input Embeddings";
            }

            if (layer === "output") {
                const vocabIdx = parseInt(cIdx);
                // Tokens are guaranteed loaded when run is loaded (see useRun.svelte.ts)
                const allTokens = (runState.allTokens as { status: "loaded"; data: TokenInfo[] }).data;
                const tokenInfo = allTokens.find((t) => t.id === vocabIdx);
                if (!tokenInfo) throw new Error(`Token not found for vocab index ${vocabIdx}`);
                return tokenInfo.string;
            }
        }

        throw new Error(`getTokenLabel called on non-token node: ${key}`);
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
            {@const formattedKey = formatNodeKeyForDisplay(key)}
            {@const isToken = isTokenNode(key)}
            {@const interp = isToken ? undefined : getInterpretation(key)}
            {@const hasInterpretation = interp?.status === "generated"}
            {@const pillLabel = hasInterpretation ? interp.data.label : formattedKey}
            <div class="pill-container" onmouseenter={(e) => handleMouseEnter(key, e)} onmouseleave={handleMouseLeave}>
                <button class="edge-pill" style="background: {bgColor};" onclick={() => onClick(key)}>
                    <span class="node-key" style="color: {textColor};">{pillLabel}</span>
                </button>
                {#if hoveredKey === key && tooltipPosition}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div
                        class="tooltip"
                        style="top: {tooltipPosition.top}px; left: {tooltipPosition.left}px;"
                        onmouseenter={() => (hoveredKey = key)}
                        onmouseleave={handleMouseLeave}
                    >
                        <div class="tooltip-value">Attribution: {value.toFixed(3)}</div>
                        {#if hasInterpretation}
                            <div class="tooltip-label">{interp.data.label}</div>
                            <button class="tooltip-node-key copyable" onclick={() => copyToClipboard(formattedKey)}>
                                {formattedKey}
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
                        {:else if isToken}
                            <div class="tooltip-token">Token: {getTokenLabel(key)}</div>
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
        flex: 0 1 auto;
        min-width: 0;
        max-width: 100%;
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
        max-width: 100%;
        overflow: hidden;
    }

    .node-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
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
        word-wrap: break-word;
    }

    .tooltip-node-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-secondary);
        margin-bottom: var(--space-1);
    }

    .tooltip-node-key.copyable {
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

    .tooltip-node-key.copyable:hover {
        background: var(--bg-surface);
    }

    .tooltip-node-key.copyable .copy-icon {
        opacity: 0.4;
        flex-shrink: 0;
    }

    .tooltip-node-key.copyable:hover .copy-icon {
        opacity: 0.8;
    }

    .tooltip-confidence {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
