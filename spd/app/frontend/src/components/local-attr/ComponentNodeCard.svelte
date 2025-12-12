<script lang="ts">
    import type {
        ComponentDetail,
        ComponentSummary,
        ComponentCorrelations,
        TokenStats,
        Edge,
        CorrelatedComponent,
    } from "../../lib/localAttributionsTypes";
    import { getComponentCorrelations, getComponentTokenStats } from "../../lib/localAttributionsApi";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "../ui/ComponentCorrelationMetrics.svelte";
    import ComponentPillList from "../ui/ComponentPillList.svelte";
    import TokenStatsSection from "../ui/TokenStatsSection.svelte";
    import SectionHeader from "../ui/SectionHeader.svelte";
    import StatusText from "../ui/StatusText.svelte";

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx: number;
        summary: ComponentSummary | null;
        edges: Edge[];
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    } & ({ detail: ComponentDetail; isLoading?: never } | { detail: null; isLoading: boolean });

    let { layer, cIdx, seqIdx, summary, edges, detail, isLoading, onPinComponent }: Props = $props();

    // Handle clicking a correlated component - parse key and pin it at same seqIdx
    function handleCorrelationClick(componentKey: string) {
        if (!onPinComponent) return;
        // componentKey format: "layer:cIdx" e.g. "h.0.attn.q_proj:5"
        const [clickedLayer, clickedCIdx] = componentKey.split(":");
        onPinComponent(clickedLayer, parseInt(clickedCIdx), seqIdx);
    }

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
        getComponentCorrelations(layer, cIdx, 1000)
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
        getComponentTokenStats(layer, cIdx, 1000)
            .then((data) => {
                tokenStats = data;
            })
            .finally(() => {
                tokenStatsLoading = false;
            });
    });

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

    // === Edge attribution lists ===
    const currentNodeKey = $derived(`${layer}:${seqIdx}:${cIdx}`);
    const N_EDGES_TO_DISPLAY = 20;

    // Convert edge to CorrelatedComponent format for ComponentPillList
    function edgeToItem(targetNodeKey: string, val: number, maxVal: number): CorrelatedComponent {
        return {
            component_key: targetNodeKey,
            score: Math.abs(val) / maxVal,
            count_i: 0,
            count_j: 0,
            count_ij: 0,
            n_tokens: 0,
        };
    }

    // Incoming edges: edges where this node is the target (what influences this node)
    const incomingPositive = $derived.by(() => {
        const filtered = edges.filter((e) => e.tgt === currentNodeKey && e.val > 0);
        const sorted = filtered.sort((a, b) => b.val - a.val).slice(0, N_EDGES_TO_DISPLAY);
        const maxVal = sorted[0]?.val || 1;
        return sorted.map((e) => edgeToItem(e.src, e.val, maxVal));
    });

    const incomingNegative = $derived.by(() => {
        const filtered = edges.filter((e) => e.tgt === currentNodeKey && e.val < 0);
        const sorted = filtered.sort((a, b) => a.val - b.val).slice(0, N_EDGES_TO_DISPLAY);
        const maxVal = Math.abs(sorted[0]?.val || 1);
        return sorted.map((e) => edgeToItem(e.src, e.val, maxVal));
    });

    // Outgoing edges: edges where this node is the source (what this node influences)
    const outgoingPositive = $derived.by(() => {
        const filtered = edges.filter((e) => e.src === currentNodeKey && e.val > 0);
        const sorted = filtered.sort((a, b) => b.val - a.val).slice(0, N_EDGES_TO_DISPLAY);
        const maxVal = sorted[0]?.val || 1;
        return sorted.map((e) => edgeToItem(e.tgt, e.val, maxVal));
    });

    const outgoingNegative = $derived.by(() => {
        const filtered = edges.filter((e) => e.src === currentNodeKey && e.val < 0);
        const sorted = filtered.sort((a, b) => a.val - b.val).slice(0, N_EDGES_TO_DISPLAY);
        const maxVal = Math.abs(sorted[0]?.val || 1);
        return sorted.map((e) => edgeToItem(e.tgt, e.val, maxVal));
    });

    const hasAnyEdges = $derived(
        incomingPositive.length > 0 ||
            incomingNegative.length > 0 ||
            outgoingPositive.length > 0 ||
            outgoingNegative.length > 0,
    );

    // Handle clicking an edge node - parse key and pin it
    function handleEdgeNodeClick(nodeKey: string) {
        if (!onPinComponent) return;
        // nodeKey format: "layer:seq:cIdx"
        const [clickedLayer, clickedSeqIdx, clickedCIdx] = nodeKey.split(":");
        onPinComponent(clickedLayer, parseInt(clickedCIdx), parseInt(clickedSeqIdx));
    }
</script>

<div class="component-node-card">
    <SectionHeader title="Position {seqIdx}" level="h4">
        {#if summary}
            <span class="mean-ci">Mean CI: {formatMeanCi(summary.mean_ci)}</span>
        {/if}
    </SectionHeader>

    <div class="token-stats-row">
        <TokenStatsSection
            sectionTitle="Input Tokens"
            sectionSubtitle="(what activates this component)"
            loading={tokenStatsLoading}
            lists={[
                {
                    title: "Top PMI",
                    mathNotation: "log(P(firing, token) / P(firing)P(token))",
                    items: inputTopPmi,
                },
            ]}
        />

        <TokenStatsSection
            sectionTitle="Output Tokens"
            sectionSubtitle="(what this component predicts)"
            loading={tokenStatsLoading}
            lists={[
                {
                    title: "Top PMI",
                    mathNotation: "positive association with predictions",
                    items: outputTopPmi,
                },
                {
                    title: "Bottom PMI",
                    mathNotation: "negative association with predictions",
                    items: outputBottomPmi,
                },
            ]}
        />
    </div>

    <ComponentProbeInput {layer} componentIdx={cIdx} />

    <!-- Component correlations -->
    <div class="correlations-section">
        <SectionHeader title="Correlated Components" />
        {#if correlations}
            <ComponentCorrelationMetrics {correlations} pageSize={16} onComponentClick={handleCorrelationClick} />
        {:else if correlationsLoading}
            <StatusText>Loading...</StatusText>
        {:else}
            <StatusText>No correlations available.</StatusText>
        {/if}
    </div>

    <!-- Edge attributions (local, for this datapoint) -->
    {#if displaySettings.showEdgeAttributions && hasAnyEdges}
        <div class="edge-attributions-section">
            <SectionHeader title="Edge Attributions" />
            <div class="edge-lists-grid">
                {#if incomingPositive.length > 0 || incomingNegative.length > 0}
                    <div class="edge-list-group">
                        <h5>Incoming</h5>
                        {#if incomingPositive.length > 0}
                            <div class="edge-list">
                                <span class="edge-list-title">Most attributed</span>
                                <ComponentPillList
                                    items={incomingPositive}
                                    pageSize={10}
                                    onComponentClick={handleEdgeNodeClick}
                                    colorScheme="green"
                                />
                            </div>
                        {/if}
                        {#if incomingNegative.length > 0}
                            <div class="edge-list">
                                <span class="edge-list-title">Least attributed</span>
                                <ComponentPillList
                                    items={incomingNegative}
                                    pageSize={10}
                                    onComponentClick={handleEdgeNodeClick}
                                    colorScheme="red"
                                />
                            </div>
                        {/if}
                    </div>
                {/if}
                {#if outgoingPositive.length > 0 || outgoingNegative.length > 0}
                    <div class="edge-list-group">
                        <h5>Outgoing</h5>
                        {#if outgoingPositive.length > 0}
                            <div class="edge-list">
                                <span class="edge-list-title">Most attributed</span>
                                <ComponentPillList
                                    items={outgoingPositive}
                                    pageSize={10}
                                    onComponentClick={handleEdgeNodeClick}
                                    colorScheme="green"
                                />
                            </div>
                        {/if}
                        {#if outgoingNegative.length > 0}
                            <div class="edge-list">
                                <span class="edge-list-title">Least attributed</span>
                                <ComponentPillList
                                    items={outgoingNegative}
                                    pageSize={10}
                                    onComponentClick={handleEdgeNodeClick}
                                    colorScheme="red"
                                />
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>
        </div>
    {/if}

    {#if detail}
        {#if detail.example_tokens.length > 0}
            <!-- Full mode: paged table with filtering -->
            <SectionHeader title="Activating Examples ({detail.example_tokens.length})" />
            <ActivationContextsPagedTable
                exampleTokens={detail.example_tokens}
                exampleCi={detail.example_ci}
                exampleActivePos={detail.example_active_pos}
                {activatingTokens}
            />
        {/if}
    {:else if isLoading}
        <StatusText>Loading details...</StatusText>
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

    .token-stats-row > :global(*) {
        flex: 1;
        min-width: 0;
    }

    .correlations-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .edge-attributions-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .edge-lists-grid {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-4);
    }

    .edge-list-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .edge-list-group h5 {
        margin: 0;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .edge-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .edge-list-title {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-style: italic;
    }
</style>
