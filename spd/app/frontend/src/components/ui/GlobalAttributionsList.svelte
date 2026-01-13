<script lang="ts">
    import { getContext } from "svelte";
    import type { AttributingComponent } from "../../lib/api/correlations";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        items: AttributingComponent[];
        onComponentClick?: (componentKey: string) => void;
        pageSize: number;
        signColor: "positive" | "negative";
    };

    let { items, onComponentClick, pageSize = 40, signColor }: Props = $props();

    function getInterpretationLabel(componentKey: string): string | null {
        const interp = runState.getInterpretation(componentKey);
        if (interp.status === "loaded" && interp.data.status === "generated") return interp.data.data.label;
        return null;
    }

    let currentPage = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

    function formatAttribution(value: number): string {
        const absVal = Math.abs(value);
        if (absVal >= 1000) return value.toExponential(1);
        if (absVal >= 1) return value.toFixed(1);
        return value.toFixed(3);
    }

    function getBorderColor(signColor: "positive" | "negative"): string {
        return signColor === "positive" ? "rgba(22, 163, 74, 0.8)" : "rgba(220, 38, 38, 0.8)";
    }
</script>

<div class="attribution-list">
    {#if totalPages > 1}
        <div class="pagination">
            <button onclick={() => currentPage--} disabled={currentPage === 0}>&lt;</button>
            <span>{currentPage + 1} / {totalPages}</span>
            <button onclick={() => currentPage++} disabled={currentPage >= totalPages - 1}>&gt;</button>
        </div>
    {/if}
    <div class="components">
        {#each paginatedItems as { component_key, attribution } (component_key)}
            {@const borderColor = getBorderColor(signColor)}
            {@const label = getInterpretationLabel(component_key)}
            <button
                class="component-pill"
                class:clickable={!!onComponentClick}
                style="border-left: 4px solid {borderColor};"
                onclick={() => onComponentClick?.(component_key)}
                title={component_key}
            >
                <div class="pill-content">
                    {#if label}
                        <span class="interp-label">{label}</span>
                    {:else}
                        <span class="component-text">{component_key}</span>
                    {/if}
                    <span class="attribution-value" class:positive={signColor === "positive"} class:negative={signColor === "negative"}>
                        {formatAttribution(attribution)}
                    </span>
                </div>
            </button>
        {/each}
    </div>
</div>

<style>
    .attribution-list {
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
        white-space: nowrap;
        cursor: default;
        position: relative;
        border: none;
        background: var(--bg-surface);
        color: var(--text-primary);
        font-family: inherit;
        font-size: inherit;
        min-width: 80px;
    }

    .pill-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-1);
    }

    .component-pill.clickable {
        cursor: pointer;
    }

    .interp-label {
        font-family: var(--font-sans);
        font-weight: 500;
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .attribution-value {
        font-weight: 600;
    }

    .attribution-value.positive {
        color: rgb(22, 163, 74);
    }

    .attribution-value.negative {
        color: rgb(220, 38, 38);
    }
</style>
