<script lang="ts">
    import type { CorrelatedComponent } from "../../lib/localAttributionsTypes";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import SetOverlapVis from "./SetOverlapVis.svelte";
    import { lerp } from "../local-attr/graphUtils";

    type Props = {
        items: CorrelatedComponent[];
        onComponentClick?: (componentKey: string) => void;
        pageSize: number;
        colorScheme?: "green" | "red";
    };

    let { items, onComponentClick, pageSize = 40, colorScheme = "green" }: Props = $props();

    let currentPage = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

    // Reset page when items change
    $effect(() => {
        items; // eslint-disable-line @typescript-eslint/no-unused-expressions
        currentPage = 0;
    });

    function getBgColor(score: number): string {
        const intensity = lerp(0, 0.8, score);
        if (colorScheme === "red") {
            return `rgba(220, 38, 38, ${intensity})`;
        }
        return `rgba(22, 163, 74, ${intensity})`;
    }
</script>

<div class="component-pill-list">
    {#if totalPages > 1}
        <div class="pagination">
            <button onclick={() => currentPage--} disabled={currentPage === 0}>&lt;</button>
            <span>{currentPage + 1} / {totalPages}</span>
            <button onclick={() => currentPage++} disabled={currentPage >= totalPages - 1}>&gt;</button>
        </div>
    {/if}
    <div class="components">
        {#each paginatedItems as { component_key, score, count_i, count_j, count_ij, n_tokens } (component_key)}
            {@const bgColor = getBgColor(score)}
            {@const textColor = score > 0.8 ? "white" : "var(--text-primary)"}
            <button
                class="component-pill"
                class:clickable={!!onComponentClick}
                style="background: {bgColor};"
                onclick={() => onComponentClick?.(component_key)}
            >
                <div class="pill-content">
                    <span class="component-text" style="color: {textColor};">{component_key} {score.toFixed(2)}</span>
                </div>
                {#if displaySettings.showSetOverlapVis && n_tokens > 0}
                    <SetOverlapVis
                        countA={count_i}
                        countB={count_j}
                        countIntersection={count_ij}
                        totalCount={n_tokens}
                    />
                {/if}
            </button>
        {/each}
    </div>
</div>

<style>
    .component-pill-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .components {
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

    .component-pill {
        display: inline-flex;
        flex-direction: column;
        gap: 2px;
        padding: 4px 6px;
        border-radius: 3px;
        white-space: nowrap;
        cursor: default;
        position: relative;
        border: 1px solid var(--border-default);
        font-family: inherit;
        font-size: inherit;
        min-width: 80px;
    }

    .pill-content {
        display: flex;
        align-items: center;
        gap: var(--space-1);
    }

    .component-pill.clickable {
        cursor: pointer;
    }
</style>
