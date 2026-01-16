<script lang="ts">
    import { getContext } from "svelte";
    import type { OutputProbEntry, Edge } from "../../lib/promptAttributionsTypes";
    import { getLayerDisplayName } from "../../lib/promptAttributionsTypes";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";
    import ComponentNodeCard from "./ComponentNodeCard.svelte";
    import OutputNodeCard from "./OutputNodeCard.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type HoveredNode = {
        layer: string;
        seqIdx: number;
        cIdx: number;
    };

    type Props = {
        hoveredNode: HoveredNode;
        tooltipPos: { x: number; y: number };
        hideNodeCard?: boolean;
        outputProbs: Record<string, OutputProbEntry>;
        nodeCiVals: Record<string, number>;
        nodeSubcompActs: Record<string, number>;
        tokens: string[];
        edgesBySource: Map<string, Edge[]>;
        edgesByTarget: Map<string, Edge[]>;
        onMouseEnter: () => void;
        onMouseLeave: () => void;
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    };

    let {
        hoveredNode,
        tooltipPos,
        hideNodeCard = false,
        outputProbs,
        nodeCiVals,
        nodeSubcompActs,
        tokens,
        edgesBySource,
        edgesByTarget,
        onMouseEnter,
        onMouseLeave,
        onPinComponent,
    }: Props = $props();

    const isWte = $derived(hoveredNode.layer === "wte");
    const isOutput = $derived(hoveredNode.layer === "output");
    const isComponent = $derived(!isWte && !isOutput);

    // Get CI value for component nodes
    const ciVal = $derived.by(() => {
        if (!isComponent) return null;
        const key = `${hoveredNode.layer}:${hoveredNode.seqIdx}:${hoveredNode.cIdx}`;
        return nodeCiVals[key] ?? null;
    });

    // Get subcomponent activation for component nodes
    const subcompAct = $derived.by(() => {
        if (!isComponent) return null;
        const key = `${hoveredNode.layer}:${hoveredNode.seqIdx}:${hoveredNode.cIdx}`;
        return nodeSubcompActs[key] ?? null;
    });

    // Get cluster ID for component nodes (undefined = no mapping, null = singleton, number = cluster)
    const clusterId = $derived(
        isComponent ? runState.clusterMapping?.data[`${hoveredNode.layer}:${hoveredNode.cIdx}`] : undefined,
    );

    const token = $derived.by(() => {
        if (hoveredNode.seqIdx >= tokens.length) {
            throw new Error(
                `NodeTooltip: seqIdx ${hoveredNode.seqIdx} out of bounds for tokens length ${tokens.length}`,
            );
        }
        return tokens[hoveredNode.seqIdx];
    });
</script>

<div
    class="node-tooltip"
    style="left: {tooltipPos.x}px; top: {tooltipPos.y}px;"
    onmouseenter={onMouseEnter}
    onmouseleave={onMouseLeave}
    onwheel={(e) => e.stopPropagation()}
>
    <h3>{getLayerDisplayName(hoveredNode.layer)}:{hoveredNode.seqIdx}:{hoveredNode.cIdx}</h3>
    {#if isComponent && ciVal !== null}
        <div class="ci-value">CI: {ciVal.toFixed(3)}</div>
    {/if}
    {#if isComponent && subcompAct !== null}
        <div class="subcomp-act">Subcomp Act: {subcompAct.toFixed(3)}</div>
    {/if}
    {#if clusterId !== undefined}
        <div class="cluster-id">Cluster: {clusterId ?? "null"}</div>
    {/if}
    {#if isWte}
        <p class="wte-info">Input embedding at position {hoveredNode.seqIdx}</p>
        <div class="wte-content">
            <div class="wte-token">"{token}"</div>
            <p class="wte-stats">
                <strong>Position:</strong>
                {hoveredNode.seqIdx}
            </p>
        </div>
    {:else if isOutput}
        <OutputNodeCard cIdx={hoveredNode.cIdx} {outputProbs} seqIdx={hoveredNode.seqIdx} />
    {:else if !hideNodeCard}
        <!-- Key forces remount when component identity changes, so ComponentNodeCard can load on mount -->
        {#key `${hoveredNode.layer}:${hoveredNode.cIdx}`}
            <ComponentNodeCard
                layer={hoveredNode.layer}
                cIdx={hoveredNode.cIdx}
                seqIdx={hoveredNode.seqIdx}
                {edgesBySource}
                {edgesByTarget}
                {tokens}
                {outputProbs}
                {onPinComponent}
            />
        {/key}
    {/if}
</div>

<style>
    .node-tooltip {
        position: fixed;
        z-index: 1000;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        padding: var(--space-3);
        width: fit-content;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .ci-value {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        font-weight: 600;
        margin: var(--space-1) 0 var(--space-2) 0;
    }

    .subcomp-act {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--accent-secondary);
        font-weight: 600;
        margin: var(--space-1) 0 var(--space-2) 0;
    }

    .cluster-id {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        font-weight: 600;
        margin: var(--space-1) 0 var(--space-2) 0;
    }

    .wte-info {
        margin: var(--space-2) 0 0 0;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }
</style>
