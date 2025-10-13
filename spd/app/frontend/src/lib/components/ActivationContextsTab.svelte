<script lang="ts">
    import type { ActivationContextsConfig, SubcomponentActivationContexts } from "$lib/api";
    import * as api from "$lib/api";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";

    let loading = false;
    let allLayersData: Record<string, SubcomponentActivationContexts[]> | null = null;

    // Configuration parameters
    let importanceThreshold = 0.01;
    let maxExamplesPerSubcomponent = 100;
    let nBatches = 1;
    let batchSize = 32;
    let nTokensEitherSide = 10;

    async function loadContexts() {
        loading = true;
        allLayersData = null;
        try {
            const config: ActivationContextsConfig = {
                importance_threshold: importanceThreshold,
                max_examples_per_subcomponent: maxExamplesPerSubcomponent,
                n_batches: nBatches,
                batch_size: batchSize,
                n_tokens_either_side: nTokensEitherSide
            };
            const data = await api.getSubcomponentActivationContexts(config);
            allLayersData = data.layers;
        } catch (e) {
            if ((e as any)?.name !== "AbortError") {
                console.error(e);
            }
        } finally {
            loading = false;
        }
    }
</script>

<div class="tab-content">
    <div class="controls">
        <div class="config-section">
            <h4>Configuration</h4>
            <div class="config-grid">
                <div class="config-item">
                    <label for="importance-threshold">Importance Threshold:</label>
                    <input
                        id="importance-threshold"
                        type="number"
                        step="0.001"
                        min="0"
                        max="1"
                        bind:value={importanceThreshold}
                    />
                </div>

                <div class="config-item">
                    <label for="max-examples">Max Examples per Subcomponent:</label>
                    <input
                        id="max-examples"
                        type="number"
                        step="10"
                        min="1"
                        bind:value={maxExamplesPerSubcomponent}
                    />
                </div>

                <div class="config-item">
                    <label for="n-steps">Number of Batches:</label>
                    <input id="n-steps" type="number" step="1" min="1" bind:value={nBatches} />
                </div>

                <div class="config-item">
                    <label for="batch-size">Batch Size:</label>
                    <input id="batch-size" type="number" step="1" min="1" bind:value={batchSize} />
                </div>

                <div class="config-item">
                    <label for="n-tokens">Context Tokens Either Side:</label>
                    <input
                        id="n-tokens"
                        type="number"
                        step="1"
                        min="0"
                        bind:value={nTokensEitherSide}
                    />
                </div>
            </div>
            <button class="load-button" on:click={loadContexts} disabled={loading}>
                {loading ? "Loading..." : "Load Contexts"}
            </button>
        </div>
    </div>

    {#if loading}
        <div class="loading">Loading...</div>
    {/if}

    {#if allLayersData}
        <ActivationContextsViewer {allLayersData} />
    {/if}
</div>

<style>
    .tab-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }

    .controls {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        flex-direction: column;
    }

    .config-section {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .config-section h4 {
        margin: 0;
        font-size: 1rem;
        color: #495057;
    }

    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
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
</style>
