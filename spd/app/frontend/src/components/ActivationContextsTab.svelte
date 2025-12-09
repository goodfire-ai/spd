<script lang="ts">
    import * as api from "../lib/api";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";

    let loading = $state(false);
    let harvestMetadata = $state<api.HarvestMetadata | null>(null);
    let progress = $state<api.ProgressUpdate | null>(null);

    // Configuration parameters
    let nBatches = $state(10);
    let batchSize = $state(16);
    let nTokensEitherSide = $state(8);
    let separationTokens = $state(8);
    let importanceThreshold = $state(0.0);
    let topkExamples = $state(200);

    async function loadContexts() {
        loading = true;
        harvestMetadata = null;
        progress = null;
        try {
            const data = await api.getSubcomponentActivationContexts(
                {
                    n_batches: nBatches,
                    batch_size: batchSize,
                    n_tokens_either_side: nTokensEitherSide,
                    importance_threshold: importanceThreshold,
                    topk_examples: topkExamples,
                    separation_tokens: separationTokens,
                },
                (p) => {
                    progress = p;
                },
            );
            harvestMetadata = data;
        } catch (error) {
            console.error("Error loading contexts", error);
        } finally {
            loading = false;
            progress = null;
        }
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && !loading) {
            loadContexts();
        }
    }
</script>

<div class="tab-content">
    <div class="controls">
        <div class="config-section">
            <h4>Configuration</h4>
            <div class="config-grid">
                <div class="config-item">
                    <label for="n-steps">Number of Batches:</label>
                    <input
                        id="n-steps"
                        type="number"
                        step="1"
                        min="1"
                        bind:value={nBatches}
                        onkeydown={handleKeydown}
                    />
                </div>

                <div class="config-item">
                    <label for="batch-size">Batch Size:</label>
                    <input
                        id="batch-size"
                        type="number"
                        step="1"
                        min="1"
                        bind:value={batchSize}
                        onkeydown={handleKeydown}
                    />
                </div>

                <div class="config-item">
                    <label for="n-tokens">Context Tokens Either Side:</label>
                    <input
                        id="n-tokens"
                        type="number"
                        step="1"
                        min="0"
                        bind:value={nTokensEitherSide}
                        onkeydown={handleKeydown}
                    />
                </div>

                <div class="config-item">
                    <label for="separation-tokens">Token Separation:</label>
                    <input
                        id="separation-tokens"
                        type="number"
                        step="1"
                        min="0"
                        bind:value={separationTokens}
                        onkeydown={handleKeydown}
                    />
                </div>

                <div class="config-item">
                    <label for="importance-threshold">Importance Threshold:</label>
                    <input
                        id="importance-threshold"
                        type="number"
                        step="0.001"
                        min="0"
                        max="1"
                        bind:value={importanceThreshold}
                        onkeydown={handleKeydown}
                    />
                </div>

                <div class="config-item">
                    <label for="topk-examples">Top K Examples:</label>
                    <input
                        id="topk-examples"
                        type="number"
                        step="1"
                        min="1"
                        bind:value={topkExamples}
                        onkeydown={handleKeydown}
                    />
                </div>
            </div>
            <button class="load-button" onclick={loadContexts} disabled={loading}>
                {loading ? "Loading..." : "Load Contexts"}
            </button>
        </div>
    </div>

    {#if loading && progress}
        <div class="progress-container">
            <div class="progress-text">
                Processing... {(progress.progress * 100).toFixed(1)}%
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress.progress * 100}%"></div>
            </div>
        </div>
    {:else if loading}
        <div class="loading">Loading...</div>
    {/if}

    {#if harvestMetadata}
        <ActivationContextsViewer {harvestMetadata} />
    {/if}
</div>

<style>
    .tab-content {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding: var(--space-3);
    }

    .controls {
        display: flex;
        gap: var(--space-2);
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        flex-direction: column;
    }

    .config-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .config-section h4 {
        margin: 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: var(--space-2);
    }

    .config-item {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .config-item label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .config-item input {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
    }

    .config-item input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .load-button {
        padding: var(--space-2) var(--space-4);
        border: none;
        background: var(--accent-primary);
        color: white;
        font-weight: 500;
        align-self: flex-start;
    }

    .load-button:hover:not(:disabled) {
        background: var(--accent-primary-dim);
    }

    .load-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
    }

    .progress-container {
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
    }

    .progress-text {
        margin-bottom: var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .progress-bar {
        width: 100%;
        height: 4px;
        background: var(--border-default);
        border-radius: 2px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: var(--accent-primary);
        border-radius: 2px;
        transition: width 0.15s ease-out;
    }

    .loading {
        padding: var(--space-3);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
