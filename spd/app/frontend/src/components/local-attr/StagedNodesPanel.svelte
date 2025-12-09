<script lang="ts">
    import type { PinnedNode, ComponentDetail, ActivationContextsSummary, OutputProbEntry } from "../../lib/localAttributionsTypes";
    import ComponentDetailCard from "./ComponentDetailCard.svelte";

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

    function getTokenAtPosition(seqIdx: number): string {
        if (seqIdx >= 0 && seqIdx < tokens.length) {
            return tokens[seqIdx];
        }
        return "?";
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
                {@const cacheKey = `${node.layer}:${node.cIdx}`}
                {@const detail = componentDetailsCache[cacheKey]}
                {@const isLoading = componentDetailsLoading[cacheKey] ?? false}
                {@const summary = activationContextsSummary?.[node.layer]?.find((s) => s.subcomponent_idx === node.cIdx)}
                {@const token = getTokenAtPosition(node.seqIdx)}
                <div class="staged-item">
                    <div class="staged-header">
                        <div class="node-info">
                            <strong>{node.layer}:{node.seqIdx}:{node.cIdx}</strong>
                            <span class="token-preview">"{token}"</span>
                        </div>
                        <button class="unstage-btn" onclick={() => unstageNode(node)}>✕</button>
                    </div>

                    <ComponentDetailCard
                        layer={node.layer}
                        cIdx={node.cIdx}
                        seqIdx={node.seqIdx}
                        {detail}
                        {isLoading}
                        {outputProbs}
                        {summary}
                        compact
                    />
                </div>
            {/each}
        </div>
    </div>
{/if}

<style>
    .staged-container {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-top: none;
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
        min-width: 300px;
        max-width: 400px;
        border: 1px solid var(--border-default);
        padding: var(--space-3);
        background: var(--bg-elevated);
    }

    .staged-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: var(--space-2);
    }

    .node-info {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .staged-header strong {
        font-family: var(--font-mono);
        font-size: var(--text-base);
        color: var(--accent-primary);
        font-weight: 600;
    }

    .token-preview {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .unstage-btn {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
        padding: var(--space-1) var(--space-2);
    }

    .unstage-btn:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }
</style>
