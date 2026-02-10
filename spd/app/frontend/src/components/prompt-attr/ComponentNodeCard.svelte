<script lang="ts">
    import { getContext, onMount } from "svelte";
    import { computeMaxAbsComponentAct } from "../../lib/colors";
    import { COMPONENT_CARD_CONSTANTS } from "../../lib/componentCardConstants";
    import { anyCorrelationStatsEnabled, displaySettings } from "../../lib/displaySettings.svelte";
    import type { EdgeAttribution, EdgeData, OutputProbability } from "../../lib/promptAttributionsTypes";
    import { getLayerDisplayName } from "../../lib/promptAttributionsTypes";
    import { useComponentDataExpectCached } from "../../lib/useComponentDataExpectCached.svelte";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "../ui/ComponentCorrelationMetrics.svelte";
    import DatasetAttributionsSection from "../ui/DatasetAttributionsSection.svelte";
    import EdgeAttributionGrid from "../ui/EdgeAttributionGrid.svelte";
    import InterpretationBadge from "../ui/InterpretationBadge.svelte";
    import SectionHeader from "../ui/SectionHeader.svelte";
    import StatusText from "../ui/StatusText.svelte";
    import TokenStatsSection from "../ui/TokenStatsSection.svelte";

    const runState = getContext<RunContext>(RUN_KEY);
    const displayNames = $derived(runState.modelInfo?.display_names ?? {});

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx: number;
        ciVal: number | null;
        subcompAct: number | null;
        token: string;
        edgesBySource: Map<string, EdgeData[]>;
        edgesByTarget: Map<string, EdgeData[]>;
        tokens: string[];
        outputProbs: Record<string, OutputProbability>;
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    };

    let {
        layer,
        cIdx,
        seqIdx,
        ciVal,
        subcompAct,
        token,
        edgesBySource,
        edgesByTarget,
        tokens,
        outputProbs,
        onPinComponent,
    }: Props = $props();

    const clusterId = $derived(runState.clusterMapping?.data[`${layer}:${cIdx}`]);
    const intruderScore = $derived(runState.getIntruderScore(`${layer}:${cIdx}`));

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
    // Reads from prefetched cache for activation contexts, correlations, token stats.
    // Dataset attributions and interpretation details are fetched on-demand.
    const componentData = useComponentDataExpectCached();

    onMount(() => {
        componentData.load(layer, cIdx);
    });

    // Derive token lists from loaded tokenStats (null if not loaded or no data)
    const inputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats.status !== "loaded" || tokenStats.data === null) return null;
        return [
            {
                title: "Top Recall",
                mathNotation: "P(token | component fires)",
                items: tokenStats.data.input.top_recall
                    .slice(0, COMPONENT_CARD_CONSTANTS.N_INPUT_TOKENS)
                    .map(([token, value]) => ({ token, value })),
                maxScale: 1,
            },
            {
                title: "Top Precision",
                mathNotation: "P(component fires | token)",
                items: tokenStats.data.input.top_precision
                    .slice(0, COMPONENT_CARD_CONSTANTS.N_INPUT_TOKENS)
                    .map(([token, value]) => ({ token, value })),
                maxScale: 1,
            },
        ];
    });

    const outputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats.status !== "loaded" || tokenStats.data === null) return null;
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
                    .slice(0, COMPONENT_CARD_CONSTANTS.N_OUTPUT_TOKENS)
                    .map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
            {
                title: "Bottom PMI",
                mathNotation: "negative association with predictions",
                items: tokenStats.data.output.bottom_pmi
                    .slice(0, COMPONENT_CARD_CONSTANTS.N_OUTPUT_TOKENS)
                    .map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
        ];
    });

    // Format mean CI or subcomponent activation for display
    function formatNumericalValue(val: number): string {
        return Math.abs(val) < 0.001 ? val.toExponential(2) : val.toFixed(3);
    }

    // === Edge attribution lists ===
    const currentNodeKey = $derived(`${layer}:${seqIdx}:${cIdx}`);
    const N_EDGES_TO_DISPLAY = 20;

    function getTopEdgeAttributions(
        edges: EdgeData[],
        isPositive: boolean,
        getKey: (e: EdgeData) => string,
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
        if (componentData.componentDetail.status !== "loaded") return 1;
        return computeMaxAbsComponentAct(componentData.componentDetail.data.example_component_acts);
    });
</script>

<div class="component-node-card">
    <div class="card-header">
        <h3 class="node-identifier">{getLayerDisplayName(layer, displayNames)}:{seqIdx}:{cIdx}</h3>
        <div class="token-display">"{token}"</div>
        <div class="header-metrics">
            {#if ciVal !== null}
                <span class="metric">CI: {formatNumericalValue(ciVal)}</span>
            {/if}
            {#if subcompAct !== null}
                <span class="metric">Subcomp Act: {formatNumericalValue(subcompAct)}</span>
            {/if}
            {#if clusterId !== undefined}
                <span class="metric">Cluster: {clusterId ?? "null"}</span>
            {/if}
            {#if componentData.componentDetail.status === "loaded"}
                <span class="metric">Mean CI: {formatNumericalValue(componentData.componentDetail.data.mean_ci)}</span>
            {/if}
            {#if intruderScore !== null}
                <span class="metric">Intruder: {Math.round(intruderScore * 100)}%</span>
            {/if}
        </div>
    </div>

    <InterpretationBadge
        interpretation={componentData.interpretation}
        interpretationDetail={componentData.interpretationDetail}
        onGenerate={componentData.generateInterpretation}
    />

    <!-- Activating examples (from harvest data) -->
    <div class="activating-examples-section">
        <SectionHeader title="Activating Examples" />
        {#if componentData.componentDetail.status === "uninitialized"}
            <StatusText>uninitialized</StatusText>
        {:else if componentData.componentDetail.status === "loading"}
            <StatusText>Loading details...</StatusText>
        {:else if componentData.componentDetail.status === "loaded"}
            {#if componentData.componentDetail.data.example_tokens.length > 0}
                <ActivationContextsPagedTable
                    exampleTokens={componentData.componentDetail.data.example_tokens}
                    exampleCi={componentData.componentDetail.data.example_ci}
                    exampleComponentActs={componentData.componentDetail.data.example_component_acts}
                    {maxAbsComponentAct}
                />
            {/if}
        {:else if componentData.componentDetail.status === "error"}
            <StatusText>Error loading details: {String(componentData.componentDetail.error)}</StatusText>
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
            pageSize={COMPONENT_CARD_CONSTANTS.PROMPT_ATTRIBUTIONS_PAGE_SIZE}
            onClick={handleEdgeNodeClick}
            {tokens}
            {outputProbs}
        />
    {/if}

    <!-- Dataset attributions  -->
    {#if componentData.datasetAttributions.status === "uninitialized"}
        <StatusText>uninitialized</StatusText>
    {:else if componentData.datasetAttributions.status === "loaded"}
        {#if componentData.datasetAttributions.data !== null}
            <DatasetAttributionsSection
                attributions={componentData.datasetAttributions.data}
                onComponentClick={handleCorrelationClick}
            />
        {:else}
            <StatusText>No dataset attributions available.</StatusText>
        {/if}
    {:else if componentData.datasetAttributions.status === "loading"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Loading...</StatusText>
        </div>
    {:else if componentData.datasetAttributions.status === "error"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Error: {String(componentData.datasetAttributions.error)}</StatusText>
        </div>
    {/if}

    <div class="token-stats-section">
        <SectionHeader title="Token Statistics" />
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
    </div>

    <!-- Component correlations -->
    {#if anyCorrelationStatsEnabled()}
        <div class="correlations-section">
            <SectionHeader title="Correlated Components" />
            {#if componentData.correlations.status === "loading"}
                <StatusText>Loading...</StatusText>
            {:else if componentData.correlations.status === "loaded" && componentData.correlations.data}
                <ComponentCorrelationMetrics
                    correlations={componentData.correlations.data}
                    pageSize={16}
                    onComponentClick={handleCorrelationClick}
                />
            {:else if componentData.correlations.status === "error"}
                <StatusText>Error loading correlations: {String(componentData.correlations.error)}</StatusText>
            {:else}
                <StatusText>No correlations available.</StatusText>
            {/if}
        </div>
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

    .card-header {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-default);
    }

    .node-identifier {
        font-size: var(--text-base);
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    .token-display {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .header-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-3);
    }

    .metric {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .token-stats-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
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
