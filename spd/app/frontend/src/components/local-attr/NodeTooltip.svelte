<script lang="ts">
    import type { ActivationContextsSummary, ComponentDetail, ComponentSummary, OutputProbEntry } from "../../lib/localAttributionsTypes";
    import ComponentNodeCard from "./ComponentNodeCard.svelte";
    import OutputNodeCard from "./OutputNodeCard.svelte";

    type HoveredNode = {
        layer: string;
        seqIdx: number;
        cIdx: number;
    };

    type Props = {
        hoveredNode: HoveredNode;
        tooltipPos: { x: number; y: number };
        activationContextsSummary: ActivationContextsSummary | null;
        componentDetailsCache: Record<string, ComponentDetail>;
        componentDetailsLoading: Record<string, boolean>;
        outputProbs: Record<string, OutputProbEntry>;
        tokens: string[];
        onMouseEnter: () => void;
        onMouseLeave: () => void;
    };

    let {
        hoveredNode,
        tooltipPos,
        activationContextsSummary,
        componentDetailsCache,
        componentDetailsLoading,
        outputProbs,
        tokens,
        onMouseEnter,
        onMouseLeave,
    }: Props = $props();

    // Returns null if: not yet loaded, layer not in harvest, or component not above threshold
    function findComponentSummary(layer: string, cIdx: number): ComponentSummary | null {
        if (!activationContextsSummary) return null;
        const layerSummaries = activationContextsSummary[layer];
        if (!layerSummaries) return null;
        return layerSummaries.find((s) => s.subcomponent_idx === cIdx) ?? null;
    }

    const isWte = $derived(hoveredNode.layer === "wte");
    const isOutput = $derived(hoveredNode.layer === "output");

    const inputToken = $derived.by(() => {
        if (!isWte) return null;
        if (hoveredNode.seqIdx >= tokens.length) {
            throw new Error(`NodeTooltip: seqIdx ${hoveredNode.seqIdx} out of bounds for tokens length ${tokens.length}`);
        }
        return tokens[hoveredNode.seqIdx];
    });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
    class="node-tooltip"
    style="left: {tooltipPos.x}px; top: {tooltipPos.y}px;"
    onmouseenter={onMouseEnter}
    onmouseleave={onMouseLeave}
>
    <h3>{hoveredNode.layer}:{hoveredNode.seqIdx}:{hoveredNode.cIdx}</h3>
    {#if isWte}
        <div class="wte-content">
            <div class="wte-token">"{inputToken}"</div>
            <p class="wte-stats">
                <strong>Position:</strong> {hoveredNode.seqIdx}
            </p>
        </div>
    {:else if isOutput}
        <OutputNodeCard cIdx={hoveredNode.cIdx} {outputProbs} seqIdx={hoveredNode.seqIdx} />
    {:else}
        {@const cacheKey = `${hoveredNode.layer}:${hoveredNode.cIdx}`}
        {@const detail = componentDetailsCache[cacheKey] ?? null}
        {@const isLoading = componentDetailsLoading[cacheKey] ?? false}
        {@const summary = findComponentSummary(hoveredNode.layer, hoveredNode.cIdx)}
        {#if detail}
            <ComponentNodeCard
                layer={hoveredNode.layer}
                cIdx={hoveredNode.cIdx}
                seqIdx={hoveredNode.seqIdx}
                {summary}
                {detail}
                compact={true}
            />
        {:else}
            <ComponentNodeCard
                layer={hoveredNode.layer}
                cIdx={hoveredNode.cIdx}
                seqIdx={hoveredNode.seqIdx}
                {summary}
                detail={null}
                {isLoading}
                compact={true}
            />
        {/if}
    {/if}
</div>

<style>
    .node-tooltip {
        position: fixed;
        z-index: 1000;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        padding: var(--space-3);
        max-width: 500px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .node-tooltip h3 {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-primary);
    }

    .wte-content {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .wte-token {
        font-family: var(--font-mono);
        font-size: var(--text-base);
        font-weight: 600;
        color: var(--text-primary);
        padding: var(--space-2);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
    }

    .wte-stats {
        margin: 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
    }

    .wte-stats strong {
        color: var(--text-muted);
        font-weight: 500;
    }
</style>
