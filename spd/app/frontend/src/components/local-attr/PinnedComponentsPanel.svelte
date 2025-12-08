<script lang="ts">
    import type { PinnedNode, ComponentDetail, OutputProbEntry } from "../../lib/localAttributionsTypes";
    import ComponentDetailCard from "./ComponentDetailCard.svelte";

    type Props = {
        pinnedNodes: PinnedNode[];
        componentDetailsCache: Record<string, ComponentDetail>;
        outputProbs: Record<string, OutputProbEntry>;
        tokens: string[];
        onPinnedNodesChange: (nodes: PinnedNode[]) => void;
        onGoToIntervention: (text: string) => void;
    };

    let { pinnedNodes, componentDetailsCache, outputProbs, tokens, onPinnedNodesChange, onGoToIntervention }: Props =
        $props();

    function clearAll() {
        onPinnedNodesChange([]);
    }

    function unpinNode(pinned: PinnedNode) {
        onPinnedNodesChange(pinnedNodes.filter((p) => p !== pinned));
    }

    function getTokenAtPosition(seqIdx: number): string {
        if (seqIdx >= 0 && seqIdx < tokens.length) {
            return tokens[seqIdx];
        }
        return "?";
    }

    function handleGoToIntervention() {
        const text = tokens.join("");
        onGoToIntervention(text);
    }
</script>

{#if pinnedNodes.length > 0}
    <div class="pinned-container">
        <div class="pinned-container-header">
            <span>Staged Nodes ({pinnedNodes.length})</span>
            <div class="header-actions">
                <button class="run-btn" onclick={handleGoToIntervention} disabled={pinnedNodes.length === 0}>
                    Run Intervention →
                </button>
                <button onclick={clearAll}>Clear all</button>
            </div>
        </div>

        <div class="pinned-items">
            {#each pinnedNodes as pinned, idx (`${pinned.layer}:${pinned.seqIdx}:${pinned.cIdx}-${idx}`)}
                {@const detail = componentDetailsCache[`${pinned.layer}:${pinned.cIdx}`]}
                {@const isLoading = !detail && pinned.layer !== "output"}
                {@const token = getTokenAtPosition(pinned.seqIdx)}
                <div class="pinned-item">
                    <div class="pinned-header">
                        <div class="node-info">
                            <strong>{pinned.layer}:{pinned.seqIdx}:{pinned.cIdx}</strong>
                            <span class="token-preview">"{token}"</span>
                        </div>
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

    .header-actions {
        display: flex;
        gap: var(--space-2);
    }

    .pinned-container-header button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .pinned-container-header button:hover:not(:disabled) {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .run-btn {
        background: var(--accent-primary) !important;
        color: white !important;
        border-color: var(--accent-primary) !important;
    }

    .run-btn:hover:not(:disabled) {
        background: var(--accent-primary-dim) !important;
    }

    .run-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
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

    .node-info {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .pinned-header strong {
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
