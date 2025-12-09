<script lang="ts">
    import type { PinnedNode, ComponentDetail, OutputProbEntry } from "../../lib/localAttributionsTypes";
    import ComponentDetailCard from "./ComponentDetailCard.svelte";

    type Props = {
        stagedNodes: PinnedNode[];
        componentDetailsCache: Record<string, ComponentDetail>;
        outputProbs: Record<string, OutputProbEntry>;
        tokens: string[];
        runningIntervention: boolean;
        onStagedNodesChange: (nodes: PinnedNode[]) => void;
        onRunIntervention: () => void;
    };

    let {
        stagedNodes,
        componentDetailsCache,
        outputProbs,
        tokens,
        runningIntervention,
        onStagedNodesChange,
        onRunIntervention,
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

    // Validation: can't run intervention with embedding (wte) or output nodes
    const hasInvalidNodes = $derived(stagedNodes.some((n) => n.layer === "wte" || n.layer === "output"));
    const canRunIntervention = $derived(stagedNodes.length > 0 && !hasInvalidNodes && !runningIntervention);
</script>

{#if stagedNodes.length > 0}
    <div class="staged-container">
        <div class="staged-container-header">
            <span>Staged Nodes ({stagedNodes.length})</span>
            <div class="header-actions">
                {#if hasInvalidNodes}
                    <span class="validation-warning">Remove wte/output nodes to run</span>
                {/if}
                <button class="run-btn" onclick={onRunIntervention} disabled={!canRunIntervention}>
                    {runningIntervention ? "Running..." : "Run Intervention"}
                </button>
                <button onclick={clearAll}>Clear all</button>
            </div>
        </div>

        <div class="staged-items">
            {#each stagedNodes as node, idx (`${node.layer}:${node.seqIdx}:${node.cIdx}-${idx}`)}
                {@const detail = componentDetailsCache[`${node.layer}:${node.cIdx}`]}
                {@const isLoading = !detail && node.layer !== "output"}
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
    .staged-container {
        margin-top: var(--space-4);
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

    .header-actions {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .validation-warning {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--status-negative-bright);
    }

    .staged-container-header button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .staged-container-header button:hover:not(:disabled) {
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
        background: var(--status-negative);
        color: white;
        border: none;
        padding: var(--space-1) var(--space-2);
    }

    .unstage-btn:hover {
        background: var(--status-negative-bright);
    }
</style>
