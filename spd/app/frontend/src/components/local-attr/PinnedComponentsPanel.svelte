<script lang="ts">
    import type { PinnedNode, ComponentDetail, OutputProbEntry } from "../../lib/localAttributionsTypes";
    import ComponentDetailCard from "./ComponentDetailCard.svelte";

    type Props = {
        pinnedNodes: PinnedNode[];
        componentDetailsCache: Record<string, ComponentDetail>;
        outputProbs: Record<string, OutputProbEntry>;
        onPinnedNodesChange: (nodes: PinnedNode[]) => void;
    };

    let { pinnedNodes, componentDetailsCache, outputProbs, onPinnedNodesChange }: Props = $props();

    function clearAll() {
        onPinnedNodesChange([]);
    }

    function unpinNode(pinned: PinnedNode) {
        onPinnedNodesChange(pinnedNodes.filter((p) => p !== pinned));
    }
</script>

{#if pinnedNodes.length > 0}
    <div class="pinned-container">
        <div class="pinned-container-header">
            <span>Pinned Components ({pinnedNodes.length})</span>
            <button onclick={clearAll}>Clear all</button>
        </div>
        <div class="pinned-items">
            {#each pinnedNodes as pinned (`${pinned.layer}:${pinned.cIdx}`)}
                {@const detail = componentDetailsCache[`${pinned.layer}:${pinned.cIdx}`]}
                {@const isLoading = !detail && pinned.layer !== "output"}
                <div class="pinned-item">
                    <div class="pinned-header">
                        <strong>{pinned.layer}:{pinned.cIdx}</strong>
                        <button class="unpin-btn" onclick={() => unpinNode(pinned)}>✕</button>
                    </div>

                    <ComponentDetailCard
                        layer={pinned.layer}
                        cIdx={pinned.cIdx}
                        {detail}
                        {isLoading}
                        {outputProbs}
                        compact
                    />
                </div>
            {/each}
        </div>
    </div>
{/if}

<style>
    .pinned-container {
        margin-top: var(--space-4);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .pinned-container-header {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        font-weight: 600;
        color: var(--text-secondary);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
    }

    .pinned-container-header button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .pinned-container-header button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .pinned-items {
        display: flex;
        flex-direction: row;
        gap: var(--space-3);
        overflow-x: auto;
    }

    .pinned-item {
        flex-shrink: 0;
        min-width: 300px;
        max-width: 400px;
        border: 1px solid var(--border-default);
        padding: var(--space-3);
        background: var(--bg-elevated);
    }

    .pinned-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .pinned-header strong {
        font-family: var(--font-mono);
        font-size: var(--text-base);
        color: var(--accent-primary);
        font-weight: 600;
    }

    .unpin-btn {
        background: var(--status-negative);
        color: white;
        border: none;
        padding: var(--space-1) var(--space-2);
    }

    .unpin-btn:hover {
        background: var(--status-negative-bright);
    }
</style>
