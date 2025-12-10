<script lang="ts">
    import type { ComponentDetail, ComponentSummary, ComponentCorrelations } from "../../lib/localAttributionsTypes";
    import { getComponentCorrelations } from "../../lib/localAttributionsApi";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import TokenHighlights from "../TokenHighlights.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";
    import CorrelationTable from "./CorrelationTable.svelte";

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx: number;
        summary: ComponentSummary | null;
        compact: boolean;
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    } & ({ detail: ComponentDetail; isLoading?: never } | { detail: null; isLoading: boolean });

    let { layer, cIdx, seqIdx, summary, compact, detail, isLoading, onPinComponent }: Props = $props();

    // Handle clicking a correlated component - parse key and pin it at same seqIdx
    function handleCorrelationClick(componentKey: string) {
        if (!onPinComponent) return;
        // componentKey format: "layer:cIdx" e.g. "h.0.attn.q_proj:5"
        const [clickedLayer, clickedCIdx] = componentKey.split(":");
        onPinComponent(clickedLayer, parseInt(clickedCIdx), seqIdx);
    }

    // Expandable state for compact mode
    let expanded = $state(false);

    // Correlations state
    let correlations = $state<ComponentCorrelations | null>(null);
    let correlationsLoading = $state(false);

    // Fetch correlations when component changes
    $effect(() => {
        correlations = null;
        correlationsLoading = true;
        getComponentCorrelations(layer, cIdx, 10)
            .then((data) => {
                correlations = data;
            })
            .finally(() => {
                correlationsLoading = false;
            });
    });

    const TOK_CORRELATION = 10;

    const COMPACT_MAX_EXAMPLES = 5;

    // Show full paged table when not compact, or when compact but expanded
    let showFullTable = $derived(!compact || expanded);

    // Token precisions (already sorted by backend)
    const tokenPrecisionsSorted = $derived.by(() => {
        if (!detail || detail.top_precision.length === 0) return [];
        return detail.top_precision.map(([token, precision]) => ({ token, precision }));
    });
</script>

<div class="component-node-card" class:compact>
    <p class="stats">
        <strong>Position:</strong>
        {seqIdx}
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
                    activatingTokens={detail.top_recall.map(([t]) => t)}
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

        <!-- Component correlations -->
        {#if correlations}
            <h4>Correlated Components</h4>
            <div class="correlations-grid">
                <CorrelationTable
                    title="Precision"
                    mathNotation="P(that | this)"
                    items={correlations.precision}
                    maxItems={TOK_CORRELATION}
                    onComponentClick={handleCorrelationClick}
                />
                <CorrelationTable
                    title="Recall"
                    mathNotation="P(this | that)"
                    items={correlations.recall}
                    maxItems={TOK_CORRELATION}
                    onComponentClick={handleCorrelationClick}
                />
                <CorrelationTable
                    title="F1"
                    items={correlations.f1}
                    maxItems={TOK_CORRELATION}
                    onComponentClick={handleCorrelationClick}
                />
                <CorrelationTable
                    title="Jaccard"
                    items={correlations.jaccard}
                    maxItems={TOK_CORRELATION}
                    onComponentClick={handleCorrelationClick}
                />
            </div>
        {:else if correlationsLoading}
            <p class="loading-text">Loading correlations...</p>
        {/if}
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
        /* Don't let examples expand the card width */
        max-width: 0;
        min-width: 100%;
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

    .correlations-grid {
        display: grid;
        grid-template-columns: auto auto;
        gap: var(--space-3);
        margin-top: var(--space-2);
    }
</style>
