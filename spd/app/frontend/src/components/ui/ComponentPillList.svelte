<script lang="ts">
    import { getTokenHighlightBg } from "../../lib/colors";
    import type { CorrelatedComponent } from "../../lib/localAttributionsTypes";
    import { viewSettings } from "../../lib/viewSettings.svelte";
    import SetOverlapVis from "./SetOverlapVis.svelte";

    type Props = {
        items: CorrelatedComponent[];
        onComponentClick?: (componentKey: string) => void;
    };

    let { items, onComponentClick }: Props = $props();
</script>

<div class="components">
    {#each items as { component_key, score, count_i, count_j, count_ij, n_tokens } (component_key)}
        <button
            class="component-pill"
            class:clickable={!!onComponentClick}
            style="background: {getTokenHighlightBg(score)}"
            title="{component_key}: {score.toFixed(3)}"
            onclick={() => onComponentClick?.(component_key)}
        >
            <div class="pill-content">
                <span class="component-key">{component_key}</span>
                <span class="component-score">{score.toFixed(2)}</span>
            </div>
            {#if viewSettings.showSetOverlapVis}
                <SetOverlapVis countA={count_i} countB={count_j} countIntersection={count_ij} totalCount={n_tokens} />
            {/if}
        </button>
    {/each}
</div>

<style>
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

    .component-pill {
        display: inline-flex;
        flex-direction: column;
        gap: 2px;
        padding: 4px 6px;
        border-radius: 3px;
        white-space: nowrap;
        cursor: default;
        position: relative;
        border: none;
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

    .component-pill.clickable:hover {
        filter: brightness(0.9);
    }

    .component-key {
        color: var(--text-primary);
    }

    .component-score {
        color: var(--text-secondary);
        opacity: 0.8;
    }

    .component-pill::after {
        content: attr(title);
        position: absolute;
        bottom: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: 2px 6px;
        font-size: var(--text-xs);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 100;
        border-radius: 3px;
    }

    .component-pill:hover::after {
        opacity: 1;
    }
</style>
