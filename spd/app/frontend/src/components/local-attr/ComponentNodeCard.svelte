<script lang="ts">
    import type { ComponentDetail, ComponentSummary } from "../../lib/localAttributionsTypes";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import TokenHighlights from "../TokenHighlights.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx?: number;
        summary: ComponentSummary | null;
        compact: boolean;
    } & (
        | { detail: ComponentDetail; isLoading?: never }
        | { detail: null; isLoading: boolean }
    );

    let { layer, cIdx, seqIdx, summary, compact, detail, isLoading }: Props = $props();

    // Expandable state for compact mode
    let expanded = $state(false);

    const COMPACT_MAX_EXAMPLES = 5;

    // Show full paged table when not compact, or when compact but expanded
    let showFullTable = $derived(!compact || expanded);

    // Token precisions sorted by precision descending (backend sorts by recall)
    const tokenPrecisionsSorted = $derived.by(() => {
        if (!detail || detail.pr_tokens.length === 0) return [];
        const indices = detail.pr_tokens.map((_, i) => i);
        indices.sort((a, b) => detail.pr_precisions[b] - detail.pr_precisions[a]);
        return indices.map((i) => ({ token: detail.pr_tokens[i], precision: detail.pr_precisions[i] }));
    });
</script>

<div class="component-node-card" class:compact>
    <p class="stats">
        {#if seqIdx !== undefined}
            <strong>Position:</strong>
            {seqIdx}
        {/if}
        {#if summary}
            {#if seqIdx !== undefined}|{/if}
            <strong>Mean CI:</strong>
            {summary.mean_ci.toFixed(4)}
        {/if}
    </p>

    <ComponentProbeInput {layer} componentIdx={cIdx} />

    {#if detail}
        {#if detail.example_tokens.length > 0}
            {#if showFullTable}
                <!-- Full mode: paged table with filtering -->
                <div class="examples-header">
                    <h4>Activating Examples ({detail.example_tokens.length})</h4>
                    {#if compact}
                        <button class="collapse-btn" onclick={() => (expanded = false)}>Collapse</button>
                    {/if}
                </div>
                <ActivationContextsPagedTable
                    exampleTokens={detail.example_tokens}
                    exampleCi={detail.example_ci}
                    exampleActivePos={detail.example_active_pos}
                    activatingTokens={detail.pr_tokens}
                />
            {:else}
                <!-- Compact collapsed mode: simple inline examples -->
                <h4>Top Activating Examples</h4>
                <div class="examples-scroll-container">
                    {#each detail.example_tokens.slice(0, COMPACT_MAX_EXAMPLES) as tokens, i (i)}
                        <div class="example-row">
                            <TokenHighlights
                                tokenStrings={tokens}
                                tokenCi={detail.example_ci[i]}
                                activePosition={detail.example_active_pos[i]}
                            />
                        </div>
                    {/each}
                </div>
                {#if detail.example_tokens.length > COMPACT_MAX_EXAMPLES}
                    <button class="expand-btn" onclick={() => (expanded = true)}>
                        Show all {detail.example_tokens.length} examples...
                    </button>
                {/if}
            {/if}
        {/if}

        <div class="tables-row">
            {#if tokenPrecisionsSorted.length > 0}
                <div>
                    <h4>
                        Token Precision
                        <span class="math-notation">P(firing | token)</span>
                    </h4>
                    <table class="data-table">
                        <tbody>
                            {#each tokenPrecisionsSorted.slice(0, 10) as { token, precision } (token)}
                                <tr>
                                    <td><code>{token}</code></td>
                                    <td>{precision.toFixed(3)}</td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            {/if}
        </div>
    {:else if isLoading}
        <p class="loading-text">Loading details...</p>
    {/if}
</div>

<style>
    .component-node-card {
        font-size: var(--text-base);
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .component-node-card.compact {
        font-size: var(--text-sm);
    }

    .stats {
        margin: var(--space-1) 0;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }

    .stats strong {
        color: var(--text-muted);
        font-weight: 500;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
    }

    h4 {
        margin: var(--space-3) 0 var(--space-1) 0;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-weight: 600;
        letter-spacing: 0.05em;
        font-family: var(--font-sans);
    }

    .compact h4 {
        margin: var(--space-2) 0 var(--space-1) 0;
        font-size: var(--text-xs);
    }

    .examples-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-2);
    }

    .examples-header h4 {
        margin: 0;
    }

    .examples-scroll-container {
        overflow-x: auto;
        overflow-y: clip;
        scrollbar-width: none;
    }

    .examples-scroll-container::-webkit-scrollbar {
        display: none;
    }

    .example-row {
        margin: var(--space-2) 0;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        white-space: nowrap;
    }

    .expand-btn,
    .collapse-btn {
        font-size: var(--text-sm);
        color: var(--accent-primary);
        background: none;
        border: none;
        padding: var(--space-1) 0;
    }

    .expand-btn:hover,
    .collapse-btn:hover {
        color: var(--accent-primary-dim);
    }

    .tables-row {
        display: flex;
        gap: var(--space-4);
        flex-wrap: wrap;
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

    .loading-text {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-mono);
        letter-spacing: 0.05em;
    }
</style>
