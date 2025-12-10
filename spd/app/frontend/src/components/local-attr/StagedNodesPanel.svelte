<script lang="ts">
    import type {
        PinnedNode,
        ComponentDetail,
        ActivationContextsSummary,
        ComponentSummary,
        OutputProbEntry,
    } from "../../lib/localAttributionsTypes";
    import ComponentNodeCard from "./ComponentNodeCard.svelte";
    import OutputNodeCard from "./OutputNodeCard.svelte";
    import NodeHeader from "./NodeHeader.svelte";

    type Props = {
        stagedNodes: PinnedNode[];
        componentDetailsCache: Record<string, ComponentDetail>;
        componentDetailsLoading: Record<string, boolean>;
        activationContextsSummary: ActivationContextsSummary | null;
        outputProbs: Record<string, OutputProbEntry>;
        tokens: string[];
        onStagedNodesChange: (nodes: PinnedNode[]) => void;
    };

    let {
        stagedNodes,
        componentDetailsCache,
        componentDetailsLoading,
        activationContextsSummary,
        outputProbs,
        tokens,
        onStagedNodesChange,
    }: Props = $props();

    function clearAll() {
        onStagedNodesChange([]);
    }

    function unstageNode(node: PinnedNode) {
        onStagedNodesChange(stagedNodes.filter((n) => n !== node));
    }

    function pinComponent(layer: string, cIdx: number, seqIdx: number) {
        const alreadyPinned = stagedNodes.some((n) => n.layer === layer && n.cIdx === cIdx && n.seqIdx === seqIdx);
        if (alreadyPinned) return;
        onStagedNodesChange([...stagedNodes, { layer, cIdx, seqIdx }]);
    }

    function getTokenAtPosition(seqIdx: number): string {
        if (seqIdx < 0 || seqIdx >= tokens.length) {
            throw new Error(`StagedNodesPanel: seqIdx ${seqIdx} out of bounds for tokens length ${tokens.length}`);
        }
        return tokens[seqIdx];
    }

    // Returns null if: not yet loaded, layer not in harvest, or component not above threshold
    function findComponentSummary(layer: string, cIdx: number): ComponentSummary | null {
        if (!activationContextsSummary) return null;
        const layerSummaries = activationContextsSummary[layer];
        if (!layerSummaries) return null;
        return layerSummaries.find((s) => s.subcomponent_idx === cIdx) ?? null;
    }
</script>

{#if stagedNodes.length > 0}
    <div class="staged-container">
        <div class="staged-container-header">
            <span>Pinned Components ({stagedNodes.length})</span>
            <button onclick={clearAll}>Clear all</button>
        </div>

        <div class="staged-items">
            {#each stagedNodes as node, idx (`${node.layer}:${node.seqIdx}:${node.cIdx}-${idx}`)}
                {@const token = getTokenAtPosition(node.seqIdx)}
                {@const isOutput = node.layer === "output"}
                {@const isWte = node.layer === "wte"}
                <div class="staged-item">
                    <NodeHeader
                        layer={node.layer}
                        seqIdx={node.seqIdx}
                        cIdx={node.cIdx}
                        {token}
                        onClose={() => unstageNode(node)}
                    />

                    {#if isWte}
                        <p class="wte-info">Input embedding at position {node.seqIdx}</p>
                    {:else if isOutput}
                        <OutputNodeCard cIdx={node.cIdx} {outputProbs} seqIdx={node.seqIdx} />
                    {:else}
                        {@const cacheKey = `${node.layer}:${node.cIdx}`}
                        {@const detail = componentDetailsCache[cacheKey] ?? null}
                        {@const isLoading = componentDetailsLoading[cacheKey] ?? false}
                        {@const summary = findComponentSummary(node.layer, node.cIdx)}
                        {#if detail}
                            <ComponentNodeCard
                                layer={node.layer}
                                cIdx={node.cIdx}
                                seqIdx={node.seqIdx}
                                {summary}
                                {detail}
                                compact={true}
                                onPinComponent={pinComponent}
                            />
                        {:else}
                            <ComponentNodeCard
                                layer={node.layer}
                                cIdx={node.cIdx}
                                seqIdx={node.seqIdx}
                                {summary}
                                detail={null}
                                {isLoading}
                                compact={true}
                                onPinComponent={pinComponent}
                            />
                        {/if}
                    {/if}
                </div>
            {/each}
        </div>
    </div>
{/if}

<style>
    .staged-container {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .staged-container-header {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        font-weight: 600;
        color: var(--text-secondary);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
    }

    .staged-container-header button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .staged-container-header button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .staged-items {
        display: flex;
        flex-direction: row;
        gap: var(--space-3);
        overflow-x: auto;
    }

    .staged-item {
        flex-shrink: 0;
        width: fit-content;
        max-width: 800px;
        border: 1px solid var(--border-default);
        padding: var(--space-3);
        background: var(--bg-elevated);
    }

    .wte-info {
        margin: var(--space-2) 0 0 0;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }
</style>
