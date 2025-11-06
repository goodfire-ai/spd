<script lang="ts">
    import * as api from "../lib/api";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";

    let loading = $state(false);
    let harvestMetadata = $state<api.HarvestMetadata | null>(null);
    let progress = $state<api.ProgressUpdate | null>(null);

    // Configuration parameters
    let nBatches = $state(1);
    let batchSize = $state(1);
    let nTokensEitherSide = $state(10);
    let importanceThreshold = $state(0.0);
    let topkExamples = $state(40);

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
                Processing batch {progress.current + 1} of {progress.total}
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {((progress.current + 1) / progress.total) * 100}%"></div>
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
        gap: 0.5rem;
    }

    .controls {
        display: flex;
        gap: 0.5rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        flex-direction: column;
    }

    .config-section {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .config-section h4 {
        margin: 0;
        font-size: 1rem;
        color: #495057;
    }

    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.5rem;
    }

    .config-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .config-item label {
        font-size: 0.875rem;
        color: #495057;
        font-weight: 500;
    }

    .config-item input {
        padding: 0.5rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        font-size: 0.9rem;
        background: white;
    }

    .load-button {
        padding: 0.75rem 1.5rem;
        border: 1px solid #0d6efd;
        border-radius: 4px;
        background: #0d6efd;
        color: white;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500;
        transition: background 0.2s;
        align-self: flex-start;
    }

    .load-button:hover:not(:disabled) {
        background: #0b5ed7;
    }

    .load-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }

    .progress-container {
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .progress-text {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #495057;
        font-weight: 500;
    }

    .progress-bar {
        width: 100%;
        height: 24px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #0d6efd 0%, #0b5ed7 100%);
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 0.5rem;
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .loading {
        padding: 0.5rem;
        text-align: center;
        color: #6c757d;
    }
</style>
