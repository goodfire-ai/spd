<script lang="ts">
    import type { ComponentDetail, OutputProbEntry, ComponentSummary } from "../../lib/localAttributionsTypes";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import TokenHighlights from "../TokenHighlights.svelte";

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
            <div
                class="output-header"
                style="background: linear-gradient(90deg, rgba(76, 175, 80, {Math.min(
                    0.8,
                    outputProbEntry.prob + 0.1,
                )}) 0%, rgba(76, 175, 80, 0.1) 100%);"
            >
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
                    {#each detail.example_tokens.slice(0, COMPACT_MAX_EXAMPLES) as tokens, i (i)}
                        <div class="example-row">
                            <TokenHighlights
                                tokenStrings={tokens}
                                tokenCi={detail.example_ci[i]}
                                activePosition={detail.example_active_pos[i]}
                            />
                        </div>
                    {/each}
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
                        <h4>Top Input Tokens</h4>
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

                {#if detail.predicted_tokens?.length}
                    <div>
                        <h4>Top Predicted</h4>
                        <table class="data-table">
                            <tbody>
                                {#each detail.predicted_tokens.slice(0, 10) as token, i (i)}
                                    <tr>
                                        <td><code>{token}</code></td>
                                        <td>{detail.predicted_probs?.[i]?.toFixed(3)}</td>
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
    {/if}
</div>

<style>
    .component-detail-card {
        font-size: 0.9rem;
    }

    .component-detail-card.compact {
        font-size: 0.85rem;
    }

    .output-header {
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }

    .output-token {
        font-size: 1.2em;
        font-weight: bold;
    }

    .output-prob {
        font-size: 1em;
        color: #333;
    }

    .stats {
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }

    h4 {
        margin: 0.75rem 0 0.25rem 0;
        font-size: 0.85rem;
        color: #333;
    }

    .compact h4 {
        margin: 0.5rem 0 0.25rem 0;
        font-size: 0.8rem;
    }

    .examples-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
    }

    .examples-header h4 {
        margin: 0;
    }

    .example-row {
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }

    .expand-btn,
    .collapse-btn {
        font-size: 0.75rem;
        color: #1976d2;
        background: none;
        border: none;
        padding: 0.25rem 0;
        cursor: pointer;
        text-decoration: underline;
    }

    .expand-btn:hover,
    .collapse-btn:hover {
        color: #1565c0;
    }

    .tables-row {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
    }

    .data-table {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .data-table td {
        padding: 0.15rem 0.5rem;
    }

    .loading-text {
        font-size: 0.85rem;
        color: #666;
    }
</style>
