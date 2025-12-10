<script lang="ts">
    import { getTokenHighlightBg } from "../../lib/colors";
    import type { CorrelatedComponent } from "../../lib/localAttributionsTypes";

    type Props = {
        title: string;
        mathNotation?: string;
        items: CorrelatedComponent[];
        maxItems?: number;
        onComponentClick?: (componentKey: string) => void;
    };

    let { title, mathNotation, items, maxItems = 10, onComponentClick }: Props = $props();
</script>

{#if items.length > 0}
    <div class="correlation-column">
        <h5>
            {title}
            {#if mathNotation}
                <span class="math-notation">{mathNotation}</span>
            {/if}
        </h5>
        <table class="data-table correlation-table">
            <tbody>
                {#each items.slice(0, maxItems) as { component_key, score } (component_key)}
                    <tr
                        class:clickable={!!onComponentClick}
                        onclick={() => onComponentClick?.(component_key)}
                    >
                        <td><code>{component_key}</code></td>
                        <td style="background-color: {getTokenHighlightBg(score)}">{score.toFixed(3)}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    </div>
{/if}

<style>
    .correlation-column h5 {
        margin: 0 0 var(--space-1) 0;
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .correlation-column h5 .math-notation {
        font-weight: 400;
        font-style: italic;
        color: var(--text-muted);
        margin-left: var(--space-1);
    }

    .data-table {
        font-size: var(--text-sm);
        margin-top: var(--space-1);
        font-family: var(--font-mono);
        border-collapse: collapse;
    }

    .data-table td {
        padding: var(--space-1) var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .data-table td:first-child {
        color: var(--text-primary);
    }

    .data-table td:last-child {
        color: var(--text-secondary);
        text-align: right;
    }

    .data-table code {
        color: var(--text-primary);
        font-family: var(--font-mono);
    }

    .correlation-table {
        font-size: var(--text-xs);
    }

    .correlation-table td:first-child {
        white-space: nowrap;
    }

    tr.clickable {
        cursor: pointer;
    }

    tr.clickable:hover {
        background: var(--bg-inset);
    }
</style>
