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
        <h3>
            <span>Pinned Components ({pinnedNodes.length})</span>
            <button onclick={clearAll}>Clear all</button>
        </h3>
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

    .pinned-container h3 {
        margin: 0 0 var(--space-3) 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: var(--space-2);
    }

    .pinned-container h3 button {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        padding: var(--space-1) var(--space-2);
        cursor: pointer;
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .pinned-container h3 button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .pinned-items {
        display: flex;
        flex-direction: row;
        gap: var(--space-3);
        overflow-x: auto;
        padding-bottom: var(--space-2);
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
        margin-bottom: var(--space-2);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .pinned-header strong {
        font-family: var(--font-mono);
        font-size: var(--text-base);
        color: var(--accent-warm);
        font-weight: 600;
    }

    .unpin-btn {
        cursor: pointer;
        background: var(--status-negative);
        color: var(--text-primary);
        border: none;
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .unpin-btn:hover {
        background: var(--status-negative-bright);
    }
</style>
