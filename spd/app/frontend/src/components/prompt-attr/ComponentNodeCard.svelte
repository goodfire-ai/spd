<script lang="ts">
    import { onMount } from "svelte";
    import { computeMaxAbsComponentAct } from "../../lib/colors";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import { useComponentData } from "../../lib/useComponentData.svelte";
    import type { Edge, EdgeAttribution, OutputProbEntry } from "../../lib/promptAttributionsTypes";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "../ui/ComponentCorrelationMetrics.svelte";
    import DatasetAttributionsSection from "../ui/DatasetAttributionsSection.svelte";
    import EdgeAttributionGrid from "../ui/EdgeAttributionGrid.svelte";
    import InterpretationBadge from "../ui/InterpretationBadge.svelte";
    import SectionHeader from "../ui/SectionHeader.svelte";
    import StatusText from "../ui/StatusText.svelte";
    import TokenStatsSection from "../ui/TokenStatsSection.svelte";

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx: number;
        edgesBySource: Map<string, Edge[]>;
        edgesByTarget: Map<string, Edge[]>;
        tokens: string[];
        outputProbs: Record<string, OutputProbEntry>;
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    };

    let { layer, cIdx, seqIdx, edgesBySource, edgesByTarget, tokens, outputProbs, onPinComponent }: Props = $props();

    // Handle clicking a correlated component - parse key and pin it at same seqIdx
    function handleCorrelationClick(componentKey: string) {
        if (!onPinComponent) return;
        // componentKey format: "layer:cIdx" e.g. "h.0.attn.q_proj:5"
        const [clickedLayer, clickedCIdx] = componentKey.split(":");
        onPinComponent(clickedLayer, parseInt(clickedCIdx), seqIdx);
    }

    // Component data hook - call load() explicitly on mount.
    // Parents use {#key} or {#each} keys to remount this component when layer/cIdx change,
    // so we only need to load once on mount (no effect watching props).
    // Debounce to avoid bombarding backend when hovering over many nodes quickly.
    const componentData = useComponentData();
    const HOVER_DEBOUNCE_MS = 200;

    onMount(() => {
        const timeout = setTimeout(() => {
            componentData.load(layer, cIdx);
        }, HOVER_DEBOUNCE_MS);
        return () => clearTimeout(timeout);
    });

    const N_TOKENS_TO_DISPLAY_INPUT = 50;
    const N_TOKENS_TO_DISPLAY_OUTPUT = 15;

    // Derive token lists from loaded tokenStats (null if not loaded or no data)
    const inputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats?.status !== "loaded" || tokenStats.data === null) return null;
        return [
            {
                title: "Top Recall",
                mathNotation: "P(token | component fires)",
                items: tokenStats.data.input.top_recall
                    .slice(0, N_TOKENS_TO_DISPLAY_INPUT)
                    .map(([token, value]) => ({ token, value })),
                maxScale: 1,
            },
            {
                title: "Top Precision",
                mathNotation: "P(component fires | token)",
                items: tokenStats.data.input.top_precision
                    .slice(0, N_TOKENS_TO_DISPLAY_INPUT)
                    .map(([token, value]) => ({ token, value })),
                maxScale: 1,
            },
        ];
    });

    const outputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats?.status !== "loaded" || tokenStats.data === null) return null;
        // Compute max absolute PMI for scaling
        const maxAbsPmi = Math.max(
            tokenStats.data.output.top_pmi[0]?.[1] ?? 0,
            Math.abs(tokenStats.data.output.bottom_pmi?.[0]?.[1] ?? 0),
        );
        return [
            {
                title: "Top PMI",
                mathNotation: "positive association with predictions",
                items: tokenStats.data.output.top_pmi
                    .slice(0, N_TOKENS_TO_DISPLAY_OUTPUT)
                    .map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
            {
                title: "Bottom PMI",
                mathNotation: "negative association with predictions",
                items: tokenStats.data.output.bottom_pmi
                    .slice(0, N_TOKENS_TO_DISPLAY_OUTPUT)
                    .map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
        ];
    });

    // Activating tokens from token stats (for highlighting)
    const activatingTokens = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats === null || tokenStats.status !== "loaded" || tokenStats.data === null) return [];
        return tokenStats.data.input.top_recall.map(([token]) => token);
    });

    // Format mean CI or subcomponent activation for display
    function formatNumericalValue(val: number): string {
        return Math.abs(val) < 0.001 ? val.toExponential(2) : val.toFixed(3);
    }

    // === Edge attribution lists ===
    const currentNodeKey = $derived(`${layer}:${seqIdx}:${cIdx}`);
    const N_EDGES_TO_DISPLAY = 20;

    function getTopEdgeAttributions(
        edges: Edge[],
        isPositive: boolean,
        getKey: (e: Edge) => string,
    ): EdgeAttribution[] {
        const filtered = edges.filter((e) => (isPositive ? e.val > 0 : e.val < 0));
        const sorted = filtered
            .sort((a, b) => (isPositive ? b.val - a.val : a.val - b.val))
            .slice(0, N_EDGES_TO_DISPLAY);
        const maxAbsVal = Math.abs(sorted[0]?.val || 1);
        return sorted.map((e) => ({
            key: getKey(e),
            value: e.val,
            normalizedMagnitude: Math.abs(e.val) / maxAbsVal,
        }));
    }

    const incomingPositive = $derived(
        getTopEdgeAttributions(edgesByTarget.get(currentNodeKey) ?? [], true, (e) => e.src),
    );

    const incomingNegative = $derived(
        getTopEdgeAttributions(edgesByTarget.get(currentNodeKey) ?? [], false, (e) => e.src),
    );

    const outgoingPositive = $derived(
        getTopEdgeAttributions(edgesBySource.get(currentNodeKey) ?? [], true, (e) => e.tgt),
    );

    const outgoingNegative = $derived(
        getTopEdgeAttributions(edgesBySource.get(currentNodeKey) ?? [], false, (e) => e.tgt),
    );

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

    // Compute global max absolute component act for normalization (used by both activating examples and probe)
    const maxAbsComponentAct = $derived.by(() => {
        if (componentData.componentDetail?.status !== "loaded") return 1;
        return computeMaxAbsComponentAct(componentData.componentDetail.data.example_component_acts);
    });
</script>

<div class="component-node-card">
    <SectionHeader title="Position {seqIdx}" level="h4">
        {#if componentData.componentDetail?.status === "loaded"}
            <span class="mean-ci">Mean CI: {formatNumericalValue(componentData.componentDetail.data.mean_ci)}</span>
        {/if}
    </SectionHeader>

    <InterpretationBadge
        interpretation={componentData.interpretation}
        interpretationDetail={componentData.interpretationDetail}
        onGenerate={componentData.generateInterpretation}
    />

    <!-- Activating examples (from harvest data) -->
    <div class="activating-examples-section">
        <SectionHeader title="Activating Examples" />
        {#if componentData.componentDetail?.status === "loading"}
            <StatusText>Loading details...</StatusText>
        {:else if componentData.componentDetail?.status === "loaded"}
            {#if componentData.componentDetail.data.example_tokens.length > 0}
                <ActivationContextsPagedTable
                    exampleTokens={componentData.componentDetail.data.example_tokens}
                    exampleCi={componentData.componentDetail.data.example_ci}
                    exampleComponentActs={componentData.componentDetail.data.example_component_acts}
                    {activatingTokens}
                    {maxAbsComponentAct}
                />
            {/if}
        {:else if componentData.componentDetail?.status === "error"}
            <StatusText>Error loading details: {String(componentData.componentDetail.error)}</StatusText>
        {:else}
            <StatusText>Something went wrong loading details.</StatusText>
        {/if}
    </div>

    <ComponentProbeInput {layer} componentIdx={cIdx} {maxAbsComponentAct} />

    <!-- Prompt attributions -->
    {#if displaySettings.showEdgeAttributions && hasAnyEdges}
        <EdgeAttributionGrid
            title="Prompt Attributions"
            incomingLabel="Incoming"
            outgoingLabel="Outgoing"
            {incomingPositive}
            {incomingNegative}
            {outgoingPositive}
            {outgoingNegative}
            pageSize={4}
            onClick={handleEdgeNodeClick}
            {tokens}
            {outputProbs}
        />
    {/if}

    <!-- Dataset attributions  -->
    {#if componentData.datasetAttributions?.status === "loaded" && componentData.datasetAttributions.data}
        <DatasetAttributionsSection
            attributions={componentData.datasetAttributions.data}
            onComponentClick={handleCorrelationClick}
        />
    {:else if componentData.datasetAttributions?.status === "loading"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Loading...</StatusText>
        </div>
    {:else if componentData.datasetAttributions?.status === "error"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Error: {String(componentData.datasetAttributions.error)}</StatusText>
        </div>
    {/if}

    <div class="token-stats-row">
        {#if componentData.tokenStats === null || componentData.tokenStats.status === "loading"}
            <StatusText>Loading token stats...</StatusText>
        {:else if componentData.tokenStats.status === "error"}
            <StatusText>Error: {String(componentData.tokenStats.error)}</StatusText>
        {:else}
            <TokenStatsSection
                sectionTitle="Input Tokens"
                sectionSubtitle="(what activates this component)"
                lists={inputTokenLists}
            />

            <TokenStatsSection
                sectionTitle="Output Tokens"
                sectionSubtitle="(what this component predicts)"
                lists={outputTokenLists}
            />
        {/if}
    </div>

    <!-- Component correlations -->
    <div class="correlations-section">
        <SectionHeader title="Correlated Components" />
        {#if componentData.correlations?.status === "loading"}
            <StatusText>Loading...</StatusText>
        {:else if componentData.correlations?.status === "loaded" && componentData.correlations.data}
            <ComponentCorrelationMetrics
                correlations={componentData.correlations.data}
                pageSize={16}
                onComponentClick={handleCorrelationClick}
            />
        {:else if componentData.correlations?.status === "error"}
            <StatusText>Error loading correlations: {String(componentData.correlations.error)}</StatusText>
        {:else}
            <StatusText>No correlations available.</StatusText>
        {/if}
    </div>
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

    .dataset-attributions-loading {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }
</style>
