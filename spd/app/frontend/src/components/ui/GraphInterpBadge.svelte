<script lang="ts">
    import type { GraphInterpHeadline } from "../../lib/api";

    interface Props {
        headline: GraphInterpHeadline;
    }

    let { headline }: Props = $props();

    let expanded = $state(false);
</script>

<div class="graph-interp-container">
    <button class="graph-interp-badge" onclick={() => (expanded = !expanded)} type="button">
        <div class="badge-header">
            <span class="badge-label">{headline.label}</span>
            <span class="confidence confidence-{headline.confidence}">{headline.confidence}</span>
            <span class="source-tag">graph</span>
        </div>
        {#if expanded && (headline.output_label || headline.input_label)}
            <div class="sub-labels">
                {#if headline.output_label}
                    <span class="sub-label"><span class="sub-tag">out</span> {headline.output_label}</span>
                {/if}
                {#if headline.input_label}
                    <span class="sub-label"><span class="sub-tag">in</span> {headline.input_label}</span>
                {/if}
            </div>
        {/if}
    </button>
</div>

<style>
    .graph-interp-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .graph-interp-badge {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-inset);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--status-positive-bright);
        border-top: none;
        border-right: none;
        border-bottom: none;
        cursor: pointer;
        text-align: left;
        font: inherit;
        width: 100%;
    }

    .graph-interp-badge:hover {
        background: var(--bg-surface);
    }

    .badge-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .badge-label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: var(--text-sm);
    }

    .confidence {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        text-transform: uppercase;
        font-weight: 600;
    }

    .confidence-high {
        background: color-mix(in srgb, var(--status-positive-bright) 20%, transparent);
        color: var(--status-positive-bright);
    }

    .confidence-medium {
        background: color-mix(in srgb, var(--status-warning) 20%, transparent);
        color: var(--status-warning);
    }

    .confidence-low {
        background: color-mix(in srgb, var(--text-muted) 20%, transparent);
        color: var(--text-muted);
    }

    .source-tag {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--status-positive-bright) 15%, transparent);
        color: var(--status-positive-bright);
        font-weight: 600;
        text-transform: uppercase;
        margin-left: auto;
    }

    .sub-labels {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding-top: var(--space-1);
    }

    .sub-label {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .sub-tag {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        min-width: 24px;
    }
</style>
