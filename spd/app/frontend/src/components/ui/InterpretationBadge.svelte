<script lang="ts">
    import type { Interpretation } from "../../lib/localAttributionsApi";

    interface Props {
        interpretation: Interpretation | null;
        loading?: boolean;
    }

    let { interpretation, loading = false }: Props = $props();
</script>

{#if interpretation}
    <div class="interpretation-badge" title={interpretation.reasoning}>
        <span class="interpretation-label">{interpretation.label}</span>
        <span class="confidence confidence-{interpretation.confidence}">{interpretation.confidence}</span>
    </div>
{:else if loading}
    <div class="interpretation-badge loading">
        <span class="interpretation-label">Loading interpretation...</span>
    </div>
{/if}

<style>
    .interpretation-badge {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--color-accent, #6366f1);
    }

    .interpretation-badge.loading {
        opacity: 0.5;
        border-left-color: var(--text-muted);
    }

    .interpretation-label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: var(--text-sm);
    }

    .confidence {
        font-size: var(--text-xs);
        padding: 2px 6px;
        border-radius: var(--radius-sm);
        text-transform: uppercase;
        font-weight: 600;
    }

    .confidence-high {
        background: color-mix(in srgb, #22c55e 20%, transparent);
        color: #22c55e;
    }

    .confidence-medium {
        background: color-mix(in srgb, #eab308 20%, transparent);
        color: #eab308;
    }

    .confidence-low {
        background: color-mix(in srgb, var(--text-muted) 20%, transparent);
        color: var(--text-muted);
    }
</style>
