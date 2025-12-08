<script lang="ts">
    import * as api from "../lib/api";
    import type { InterventionNode, InterventionResponse } from "../lib/interventionTypes";

    type Props = {
        stagedNodes: InterventionNode[];
        onClearNodes: () => void;
        onAddNode?: (node: InterventionNode) => void;
        onRemoveNode?: (index: number) => void;
    };

    let { stagedNodes, onClearNodes, onAddNode, onRemoveNode }: Props = $props();

    let text = $state("");
    let topK = $state(10);
    let loading = $state(false);
    let error = $state<string | null>(null);
    let result = $state<InterventionResponse | null>(null);

    // Manual node entry
    let nodeEntry = $state("");
    let nodeEntryError = $state<string | null>(null);

    function parseAndAddNode() {
        nodeEntryError = null;
        const parts = nodeEntry.trim().split(":");
        if (parts.length !== 3) {
            nodeEntryError = "Format: layer:seq_pos:component_idx (e.g. h.0.attn.q_proj:2:5)";
            return;
        }
        const [layer, seqStr, cIdxStr] = parts;
        const seq_pos = parseInt(seqStr, 10);
        const component_idx = parseInt(cIdxStr, 10);
        if (isNaN(seq_pos) || isNaN(component_idx)) {
            nodeEntryError = "seq_pos and component_idx must be numbers";
            return;
        }
        if (onAddNode) {
            onAddNode({ layer, seq_pos, component_idx });
        }
        nodeEntry = "";
    }

    async function runIntervention() {
        if (!text.trim() || stagedNodes.length === 0) return;

        loading = true;
        error = null;
        result = null;

        try {
            result = await api.runIntervention(text, stagedNodes, topK);
        } catch (e) {
            error = e instanceof Error ? e.message : "Unknown error";
        } finally {
            loading = false;
        }
    }

    function formatProb(prob: number): string {
        if (prob >= 0.01) return (prob * 100).toFixed(1) + "%";
        return (prob * 100).toExponential(1) + "%";
    }
</script>

<div class="intervention-tab">
    <div class="controls-section">
        <h2>Intervention Forward Pass</h2>
        <p class="description">
            Run a forward pass with only the staged nodes active. All other components are zeroed out.
        </p>

        <div class="staged-nodes">
            <h3>Staged Nodes ({stagedNodes.length})</h3>
            {#if stagedNodes.length === 0}
                <p class="empty-message">No nodes staged. Add nodes manually below.</p>
            {:else}
                <div class="node-list">
                    {#each stagedNodes as node, i (`${node.layer}:${node.seq_pos}:${node.component_idx}`)}
                        <span class="node-tag">
                            {node.layer}:{node.seq_pos}:{node.component_idx}
                            {#if onRemoveNode}
                                <button class="remove-node" onclick={() => onRemoveNode(i)}>x</button>
                            {/if}
                        </span>
                    {/each}
                </div>
                <button class="clear-button" onclick={onClearNodes}>Clear All</button>
            {/if}

            <div class="add-node-section">
                <input
                    type="text"
                    bind:value={nodeEntry}
                    placeholder="layer:seq_pos:component_idx"
                    onkeydown={(e) => e.key === "Enter" && parseAndAddNode()}
                />
                <button onclick={parseAndAddNode} disabled={!nodeEntry.trim()}>Add Node</button>
                {#if nodeEntryError}
                    <span class="node-entry-error">{nodeEntryError}</span>
                {/if}
            </div>
        </div>

        <div class="input-section">
            <label for="text-input">Input Text:</label>
            <textarea
                id="text-input"
                bind:value={text}
                placeholder="Enter text to run through the model..."
                rows={3}
            ></textarea>
        </div>

        <div class="options-row">
            <label>
                Top-K predictions:
                <input type="number" bind:value={topK} min={1} max={50} />
            </label>
        </div>

        <button
            class="run-button"
            onclick={runIntervention}
            disabled={loading || !text.trim() || stagedNodes.length === 0}
        >
            {loading ? "Running..." : "Run Intervention"}
        </button>

        {#if error}
            <div class="error-message">{error}</div>
        {/if}
    </div>

    {#if result}
        <div class="results-section">
            <h3>Results</h3>
            <div class="results-table-wrapper">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Pos</th>
                            <th>Input Token</th>
                            <th>Top Predictions (next token)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each result.input_tokens as inputToken, pos (pos)}
                            <tr>
                                <td class="pos-cell">{pos}</td>
                                <td class="input-token-cell">
                                    <span class="token">{inputToken}</span>
                                </td>
                                <td class="predictions-cell">
                                    {#each result.predictions_per_position[pos] as pred, i (pred.token_id)}
                                        <span class="prediction" class:top={i === 0}>
                                            <span class="pred-token">{pred.token}</span>
                                            <span class="pred-prob">{formatProb(pred.prob)}</span>
                                        </span>
                                    {/each}
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        </div>
    {/if}
</div>

<style>
    .intervention-tab {
        padding: var(--space-4);
        max-width: 1200px;
        margin: 0 auto;
    }

    h2 {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-xl);
        font-weight: 600;
        color: var(--text-primary);
    }

    h3 {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-base);
        font-weight: 600;
        color: var(--text-primary);
    }

    .description {
        color: var(--text-secondary);
        font-size: var(--text-sm);
        margin-bottom: var(--space-4);
    }

    .controls-section {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-4);
        margin-bottom: var(--space-4);
    }

    .staged-nodes {
        margin-bottom: var(--space-4);
        padding: var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
    }

    .empty-message {
        color: var(--text-muted);
        font-size: var(--text-sm);
        font-style: italic;
    }

    .node-list {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-1);
        margin-bottom: var(--space-2);
    }

    .node-tag {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        background: var(--accent-primary-dim);
        color: var(--accent-primary);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        display: inline-flex;
        align-items: center;
        gap: var(--space-1);
    }

    .remove-node {
        background: none;
        border: none;
        color: var(--accent-primary);
        cursor: pointer;
        padding: 0 2px;
        font-size: var(--text-xs);
        line-height: 1;
    }

    .remove-node:hover {
        color: var(--status-negative);
    }

    .add-node-section {
        margin-top: var(--space-3);
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
    }

    .add-node-section input {
        flex: 1;
        min-width: 200px;
        padding: var(--space-1) var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        color: var(--text-primary);
    }

    .add-node-section input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .add-node-section button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }

    .add-node-section button:hover:not(:disabled) {
        background: var(--accent-primary);
        color: white;
        border-color: var(--accent-primary);
    }

    .add-node-section button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .node-entry-error {
        color: var(--status-negative);
        font-size: var(--text-xs);
        width: 100%;
    }

    .clear-button {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .clear-button:hover {
        background: var(--status-negative);
        color: white;
        border-color: var(--status-negative);
    }

    .input-section {
        margin-bottom: var(--space-3);
    }

    .input-section label {
        display: block;
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: var(--space-1);
    }

    .input-section textarea {
        width: 100%;
        padding: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-primary);
        resize: vertical;
    }

    .input-section textarea:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .options-row {
        display: flex;
        gap: var(--space-4);
        margin-bottom: var(--space-3);
    }

    .options-row label {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .options-row input[type="number"] {
        width: 60px;
        padding: var(--space-1);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-primary);
    }

    .run-button {
        padding: var(--space-2) var(--space-4);
        background: var(--accent-primary);
        color: white;
        border: none;
        font-weight: 500;
    }

    .run-button:hover:not(:disabled) {
        background: var(--accent-primary-bright);
    }

    .run-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .error-message {
        margin-top: var(--space-3);
        padding: var(--space-2);
        background: var(--status-negative);
        color: white;
        font-size: var(--text-sm);
    }

    .results-section {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-4);
    }

    .results-table-wrapper {
        overflow-x: auto;
    }

    .results-table {
        width: 100%;
        border-collapse: collapse;
        font-size: var(--text-sm);
    }

    .results-table th {
        text-align: left;
        padding: var(--space-2);
        background: var(--bg-elevated);
        border-bottom: 2px solid var(--border-default);
        font-weight: 600;
        color: var(--text-secondary);
    }

    .results-table td {
        padding: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
        vertical-align: top;
    }

    .pos-cell {
        font-family: var(--font-mono);
        color: var(--text-muted);
        width: 40px;
    }

    .input-token-cell {
        width: 100px;
    }

    .token {
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        padding: 2px 4px;
        border-radius: 2px;
    }

    .predictions-cell {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-1);
    }

    .prediction {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        background: var(--bg-elevated);
        padding: 2px 6px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-subtle);
    }

    .prediction.top {
        background: var(--accent-primary-dim);
        border-color: var(--accent-primary);
    }

    .pred-token {
        color: var(--text-primary);
    }

    .pred-prob {
        color: var(--text-muted);
        font-size: 0.85em;
    }
</style>
