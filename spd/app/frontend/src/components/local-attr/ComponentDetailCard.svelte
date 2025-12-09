<script lang="ts">
    import type { ComponentDetail, OutputProbEntry, ComponentSummary } from "../../lib/localAttributionsTypes";
    import { getOutputHeaderColor } from "../../lib/colors";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import TokenHighlights from "../TokenHighlights.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx?: number; // Only for tooltip context (single position)
        detail: ComponentDetail | null;
        isLoading: boolean;
        // For output nodes
        outputProbs?: Record<string, OutputProbEntry>;
        // For non-output nodes
        summary?: ComponentSummary;
        // Display options
        compact?: boolean; // When true, starts with simplified inline view (expandable)
    };

    let { layer, cIdx, seqIdx, detail, isLoading, outputProbs, summary, compact = false }: Props = $props();

    let isOutput = $derived(layer === "output");

    // Expandable state for compact mode
    let expanded = $state(false);

    // For output nodes: get prob entry for single position or all positions
    let outputProbEntry = $derived.by(() => {
        if (!isOutput || !outputProbs) return null;
        if (seqIdx !== undefined) {
            return outputProbs[`${seqIdx}:${cIdx}`] ?? null;
        }
        return null;
    });

    // For output nodes in pinned context: find all positions where this token appears
    let allOutputPositions = $derived.by(() => {
        if (!isOutput || !outputProbs || seqIdx !== undefined) return [];
        return Object.entries(outputProbs)
            .filter(([key]) => key.endsWith(`:${cIdx}`))
            .map(([key, entry]) => ({
                seqIdx: parseInt(key.split(":")[0]),
                prob: entry.prob,
                token: entry.token,
            }))
            .sort((a, b) => b.prob - a.prob);
    });

    function escapeHtml(text: string): string {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    // Number of examples to show in compact (collapsed) mode
    const COMPACT_MAX_EXAMPLES = 5;

    // Show full paged table when not compact, or when compact but expanded
    let showFullTable = $derived(!compact || expanded);
</script>

<div class="component-detail-card" class:compact>
    {#if isOutput}
        <!-- Output node display -->
        {#if outputProbEntry}
            <div class="output-header" style="background: {getOutputHeaderColor(outputProbEntry.prob)};">
                <div class="output-token">"{escapeHtml(outputProbEntry.token)}"</div>
                <div class="output-prob">{(outputProbEntry.prob * 100).toFixed(1)}% probability</div>
            </div>
            <p class="stats">
                <strong>Position:</strong>
                {seqIdx} |
                <strong>Vocab ID:</strong>
                {cIdx}
            </p>
        {:else if allOutputPositions.length > 0}
            <!-- Pinned output node: show all positions -->
            <p><strong>"{allOutputPositions[0].token}"</strong></p>
            <table class="data-table">
                <tbody>
                    {#each allOutputPositions as pos (pos.seqIdx)}
                        <tr>
                            <td>Pos {pos.seqIdx}</td>
                            <td>{(pos.prob * 100).toFixed(2)}%</td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    {:else}
        <!-- Non-output node display -->
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
            {#if detail.example_tokens?.length > 0}
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
                {#if detail.pr_tokens?.length > 0}
                    <div>
                        <h4>
                            Token Precision
                            <span class="math-notation">P(firing | token)</span>
                        </h4>
                        <table class="data-table">
                            <tbody>
                                {#each detail.pr_tokens.slice(0, 10) as token, i (i)}
                                    <tr>
                                        <td><code>{token}</code></td>
                                        <td>{detail.pr_precisions[i]?.toFixed(3)}</td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>
                {/if}

                <!-- TODO: Re-enable token uplift after performance optimization
                {#if detail.predicted_tokens.length > 0}
                    <div>
                        <h4>Prediction uplift</h4>
                        <table class="data-table">
                            <tbody>
                                {#each detail.predicted_tokens.slice(0, 10) as token, i (i)}
                                    <tr>
                                        <td><code>{token}</code></td>
                                        <td class="lift-cell">
                                            <span class="lift-value">{detail.predicted_lifts[i].toFixed(1)}x</span>
                                            <span class="lift-detail">
                                                ({(detail.predicted_firing_probs[i] * 100).toFixed(1)}% vs {(
                                                    detail.predicted_base_probs[i] * 100
                                                ).toFixed(1)}%)
                                            </span>
                                        </td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>
                {/if}
                -->
            </div>
        {:else if isLoading}
            <p class="loading-text">Loading details...</p>
        {/if}
    {/if}
</div>

<style>
    .component-detail-card {
        font-size: var(--text-base);
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .component-detail-card.compact {
        font-size: var(--text-sm);
    }

    .output-header {
        padding: var(--space-2) var(--space-3);
        margin-bottom: var(--space-2);
        border-left: 2px solid var(--status-positive);
    }

    .output-token {
        font-size: 1.1em;
        font-weight: 600;
        font-family: var(--font-mono);
        color: var(--text-primary);
    }

    .output-prob {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-mono);
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

    .lift-cell {
        white-space: nowrap;
    }

    .lift-value {
        color: var(--text-primary);
        font-weight: 500;
    }

    .lift-detail {
        color: var(--text-muted);
        font-size: var(--text-xs);
        margin-left: var(--space-1);
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
