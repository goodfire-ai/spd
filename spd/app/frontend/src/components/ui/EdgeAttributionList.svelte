<script lang="ts">
    import type { EdgeAttribution } from "../../lib/localAttributionsTypes";
    import { formatNodeKeyForDisplay } from "../../lib/localAttributionsTypes";
    import { runState } from "../../lib/runState.svelte";
    import { lerp } from "../local-attr/graphUtils";

    type Props = {
        items: EdgeAttribution[];
        onNodeClick: (nodeKey: string) => void;
        pageSize: number;
        direction: "positive" | "negative";
    };

    let { items, onNodeClick, pageSize, direction }: Props = $props();

    // Extract component key (layer:cIdx) from node key (layer:seq:cIdx)
    function getComponentKey(nodeKey: string): string {
        const parts = nodeKey.split(":");
        return `${parts[0]}:${parts[2]}`; // layer:cIdx
    }

    function getInterpretationLabel(nodeKey: string): string | null {
        const componentKey = getComponentKey(nodeKey);
        return runState.getInterpretation(componentKey)?.label ?? null;
    }

    let currentPage = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

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
            {@const label = getInterpretationLabel(nodeKey)}
            <button
                class="edge-pill"
                style="background: {bgColor};"
                onclick={() => onNodeClick(nodeKey)}
                title={nodeKey}
            >
                {#if label}
                    <span class="interp-label" style="color: {textColor};">{label}</span>
                {:else}
                    <span class="node-key" style="color: {textColor};">{formatNodeKeyForDisplay(nodeKey)}</span>
                {/if}
                <span class="value" style="color: {textColor};">{value.toFixed(2)}</span>
            </button>
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
</style>
