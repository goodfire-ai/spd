<script lang="ts">
    import { onMount } from "svelte";
    import ActivationContexts from "./ActivationContexts.svelte";
    import type { RunPromptResponse } from "$lib/api";

    export let result: RunPromptResponse | null;

    let selectedLayer: string = "";
    let maxExamples = 5;
    let contextSize = 10;
    let threshold = 0.01;

    // Get all component indices for the selected layer
    function getComponentIndicesForLayer(layer: string): number[] {
        if (!result) return [];
        const layerData = result.layer_cis.find((lc) => lc.module === layer);
        if (!layerData) return [];

        // Collect all unique component indices across all tokens
        const componentSet = new Set<number>();
        for (const tokenCis of layerData.token_cis) {
            for (const idx of tokenCis.indices) {
                componentSet.add(idx);
            }
        }

        return Array.from(componentSet).sort((a, b) => a - b);
    }

    // Set default layer when result changes
    $: if (result && result.layer_cis.length > 0 && !selectedLayer) {
        selectedLayer = result.layer_cis[0].module;
    }

    $: componentIndices = selectedLayer ? getComponentIndicesForLayer(selectedLayer) : [];
</script>

<div class="activation-contexts-tab">
    {#if !result}
        <div class="empty-state">
            <p>No prompt loaded. Please run a prompt first to view activation contexts.</p>
        </div>
    {:else}
        <div class="controls">
            <div class="control-group">
                <label for="layer-select">Layer:</label>
                <select id="layer-select" bind:value={selectedLayer}>
                    {#each result.layer_cis as layerData}
                        <option value={layerData.module}>{layerData.module}</option>
                    {/each}
                </select>
            </div>

            <div class="control-group">
                <label for="max-examples">Examples per component:</label>
                <input id="max-examples" type="number" min="1" max="20" bind:value={maxExamples} />
            </div>

            <div class="control-group">
                <label for="context-size">Context window:</label>
                <input id="context-size" type="number" min="5" max="50" bind:value={contextSize} />
            </div>

            <div class="control-group">
                <label for="threshold">CI Threshold:</label>
                <input
                    id="threshold"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={threshold}
                />
            </div>
        </div>

        <div class="info-banner">
            <p>
                Showing activation examples for <strong>{componentIndices.length}</strong>
                components in layer <strong>{selectedLayer}</strong>
            </p>
            <p class="help-text">
                Scroll down to see examples for each component. Highlighted tokens show where the
                component activates.
            </p>
        </div>

        <div class="components-list">
            {#if componentIndices.length === 0}
                <div class="empty-components">
                    No components found with activations in this layer.
                </div>
            {:else}
                {#each componentIndices as componentId}
                    <div class="component-section">
                        <ActivationContexts {componentId} layer={selectedLayer} compact={false} />
                    </div>
                {/each}
            {/if}
        </div>
    {/if}
</div>

<style>
    .activation-contexts-tab {
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .empty-state {
        padding: 3rem 1rem;
        text-align: center;
        color: #666;
        font-style: italic;
    }

    .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .control-group {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }

    .control-group label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
    }

    .control-group select,
    .control-group input {
        padding: 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 0.9rem;
        min-width: 150px;
    }

    .control-group input[type="number"] {
        min-width: 100px;
    }

    .info-banner {
        padding: 1rem;
        background: #e7f3ff;
        border-left: 4px solid #007bff;
        border-radius: 4px;
    }

    .info-banner p {
        margin: 0.25rem 0;
        color: #004085;
    }

    .help-text {
        font-size: 0.9rem;
        color: #004085;
        opacity: 0.8;
    }

    .components-list {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        overflow-y: auto;
        padding: 0.5rem;
    }

    .component-section {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .empty-components {
        padding: 2rem;
        text-align: center;
        color: #666;
        font-style: italic;
    }
</style>
