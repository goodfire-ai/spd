<script lang="ts">
    import type {
        ComponentDetail,
        ComponentSummary,
        ComponentCorrelations,
        TokenStats,
    } from "../../lib/localAttributionsTypes";
    import { getComponentCorrelations, getComponentTokenStats } from "../../lib/localAttributionsApi";
    import { viewSettings } from "../../lib/viewSettings.svelte";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import TokenHighlights from "../TokenHighlights.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";
    import ComponentCorrelationTable from "./ComponentCorrelationTable.svelte";
    import TokenStatsSection from "../ui/TokenStatsSection.svelte";
    import SectionHeader from "../ui/SectionHeader.svelte";
    import StatusText from "../ui/StatusText.svelte";

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

    // Token stats state (from batch job)
    let tokenStats = $state<TokenStats | null>(null);
    let tokenStatsLoading = $state(false);

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

    // Fetch token stats when component changes
    $effect(() => {
        tokenStats = null;
        tokenStatsLoading = true;
        getComponentTokenStats(layer, cIdx, 100)
            .then((data) => {
                tokenStats = data;
            })
            .finally(() => {
                tokenStatsLoading = false;
            });
    });

    // Show full paged table when not compact, or when compact but expanded
    let showFullTable = $derived(!compact || expanded);

    const N_TOKENS_TO_DISPLAY_INPUT = 50;
    const N_TOKENS_TO_DISPLAY_OUTPUT = 15;

    // === Input token stats (what tokens activate this component) ===
    const inputTopPmi = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.input.top_pmi.slice(0, N_TOKENS_TO_DISPLAY_INPUT).map(([token, value]) => ({ token, value }));
    });

    // === Output token stats (what tokens this component predicts) ===
    const outputTopPmi = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.output.top_pmi
            .slice(0, N_TOKENS_TO_DISPLAY_OUTPUT)
            .map(([token, value]) => ({ token, value }));
    });

    const outputBottomPmi = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.output.bottom_pmi
            .slice(0, N_TOKENS_TO_DISPLAY_OUTPUT)
            .map(([token, value]) => ({ token, value }));
    });

    // Activating tokens from token stats (for highlighting)
    const activatingTokens = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.input.top_recall.map(([token]) => token);
    });

    // Format mean CI for display
    function formatMeanCi(ci: number): string {
        return ci < 0.001 ? ci.toExponential(2) : ci.toFixed(3);
    }

    const COMPACT_MAX_EXAMPLES = 5;
    const N_CORRELATIONS_TO_DISPLAY = 10;
</script>

<div class="component-node-card" class:compact>
    <SectionHeader title="Position {seqIdx}" level="h4" {compact}>
        {#if summary}
            <span class="mean-ci">Mean CI: {formatMeanCi(summary.mean_ci)}</span>
        {/if}
    </SectionHeader>

    <div class="token-stats-row" class:compact>
        <TokenStatsSection
            sectionTitle="Input Tokens"
            sectionSubtitle="(what activates this component)"
            loading={tokenStatsLoading}
            {compact}
            lists={[
                {
                    title: "Top PMI",
                    mathNotation: "log(P(firing, token) / P(firing)P(token))",
                    items: inputTopPmi.slice(0, N_TOKENS_TO_DISPLAY_INPUT),
                },
            ]}
        />

        <TokenStatsSection
            sectionTitle="Output Tokens"
            sectionSubtitle="(what this component predicts)"
            loading={tokenStatsLoading}
            {compact}
            lists={[
                {
                    title: "Top PMI",
                    mathNotation: "positive association with predictions",
                    items: outputTopPmi.slice(0, N_TOKENS_TO_DISPLAY_OUTPUT),
                },
                {
                    title: "Bottom PMI",
                    mathNotation: "negative association with predictions",
                    items: outputBottomPmi.slice(0, N_TOKENS_TO_DISPLAY_OUTPUT),
                },
            ]}
        />
    </div>

    <ComponentProbeInput {layer} componentIdx={cIdx} />

    <!-- Component correlations -->
    <div class="correlations-section">
        <SectionHeader title="Correlated Components" {compact} />
        {#if correlations}
            <div class="correlations-grid" class:compact>
                {#if viewSettings.isCorrelationStatVisible("pmi")}
                    <ComponentCorrelationTable
                        title="PMI"
                        mathNotation="log(P(both) / P(A)P(B))"
                        items={correlations.pmi.slice(0, N_CORRELATIONS_TO_DISPLAY)}
                        {compact}
                        onComponentClick={handleCorrelationClick}
                    />
                {/if}
                {#if viewSettings.isCorrelationStatVisible("precision")}
                    <ComponentCorrelationTable
                        title="Precision"
                        mathNotation="P(that | this)"
                        items={correlations.precision.slice(0, N_CORRELATIONS_TO_DISPLAY)}
                        {compact}
                        onComponentClick={handleCorrelationClick}
                    />
                {/if}
                {#if viewSettings.isCorrelationStatVisible("recall")}
                    <ComponentCorrelationTable
                        title="Recall"
                        mathNotation="P(this | that)"
                        items={correlations.recall.slice(0, N_CORRELATIONS_TO_DISPLAY)}
                        {compact}
                        onComponentClick={handleCorrelationClick}
                    />
                {/if}
                {#if viewSettings.isCorrelationStatVisible("f1")}
                    <ComponentCorrelationTable
                        title="F1"
                        items={correlations.f1.slice(0, N_CORRELATIONS_TO_DISPLAY)}
                        {compact}
                        onComponentClick={handleCorrelationClick}
                    />
                {/if}
                {#if viewSettings.isCorrelationStatVisible("jaccard")}
                    <ComponentCorrelationTable
                        title="Jaccard"
                        items={correlations.jaccard.slice(0, N_CORRELATIONS_TO_DISPLAY)}
                        {compact}
                        onComponentClick={handleCorrelationClick}
                    />
                {/if}
            </div>
        {:else if correlationsLoading}
            <StatusText {compact}>Loading...</StatusText>
        {:else}
            <StatusText {compact}>No correlations available.</StatusText>
        {/if}
    </div>

    {#if detail}
        {#if detail.example_tokens.length > 0}
            {#if showFullTable}
                <!-- Full mode: paged table with filtering -->
                <div class="examples-header">
                    <SectionHeader title="Activating Examples ({detail.example_tokens.length})" {compact} />
                    {#if compact}
                        <button class="collapse-btn" class:compact onclick={() => (expanded = false)}>Collapse</button>
                    {/if}
                </div>
                <ActivationContextsPagedTable
                    exampleTokens={detail.example_tokens}
                    exampleCi={detail.example_ci}
                    exampleActivePos={detail.example_active_pos}
                    {activatingTokens}
                />
            {:else}
                <!-- Compact collapsed mode: simple inline examples -->
                <SectionHeader title="Top Activating Examples" {compact} />
                <div class="examples-scroll-container">
                    {#each detail.example_tokens.slice(0, COMPACT_MAX_EXAMPLES) as tokens, i (i)}
                        <div class="example-row" class:compact>
                            <TokenHighlights
                                tokenStrings={tokens}
                                tokenCi={detail.example_ci[i]}
                                activePosition={detail.example_active_pos[i]}
                            />
                        </div>
                    {/each}
                </div>
                {#if detail.example_tokens.length > COMPACT_MAX_EXAMPLES}
                    <button class="expand-btn" class:compact onclick={() => (expanded = true)}>
                        Show all {detail.example_tokens.length} examples...
                    </button>
                {/if}
            {/if}
        {/if}
    {:else if isLoading}
        <StatusText {compact}>Loading details...</StatusText>
    {/if}
</div>

<style>
    .component-node-card {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .component-node-card.compact {
        gap: var(--space-2);
    }

    .mean-ci {
        font-weight: 400;
        color: var(--text-muted);
        font-family: var(--font-mono);
        margin-left: var(--space-2);
    }

    .token-stats-row {
        display: flex;
        gap: var(--space-4);
    }

    .token-stats-row.compact {
        gap: var(--space-2);
    }

    .token-stats-row > :global(*) {
        flex: 1;
        min-width: 0;
    }

    .correlations-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .correlations-grid {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-3);
    }

    .correlations-grid.compact {
        gap: var(--space-2);
    }

    .examples-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-2);
    }

    .examples-scroll-container {
        overflow-x: auto;
        overflow-y: clip;
        scrollbar-width: none;
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

    .example-row.compact {
        font-size: var(--text-xs);
        margin: var(--space-1) 0;
    }

    .expand-btn,
    .collapse-btn {
        font-size: var(--text-sm);
        color: var(--accent-primary);
        background: none;
        border: none;
        padding: var(--space-1) 0;
        cursor: pointer;
    }

    .expand-btn.compact,
    .collapse-btn.compact {
        font-size: var(--text-xs);
    }

    .expand-btn:hover,
    .collapse-btn:hover {
        color: var(--accent-primary-dim);
    }
</style>
