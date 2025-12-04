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
        margin-top: 1rem;
    }

    .pinned-container h3 {
        margin: 0 0 0.75rem 0;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .pinned-container h3 button {
        font-size: 0.8rem;
        padding: 0.25rem 0.5rem;
        cursor: pointer;
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .pinned-items {
        display: flex;
        flex-direction: row;
        gap: 1rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }

    .pinned-item {
        flex-shrink: 0;
        min-width: 300px;
        max-width: 400px;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.75rem;
        background: #fafafa;
    }

    .pinned-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .unpin-btn {
        cursor: pointer;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
    }
</style>
