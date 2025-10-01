<script lang="ts">
    import { api, type ComponentActivationContexts } from "$lib/api";
    import { onMount } from "svelte";
    import ActivationContexts from "./ActivationContexts.svelte";

    export let availableComponentLayers: string[];

    let selectedLayer: string = availableComponentLayers[0];

    let exampleSets: ComponentActivationContexts[] | null = null;

    let loading = false;

    async function loadContexts() {
        loading = true;
        console.log(`loading contexts for layer ${selectedLayer}`);
        const result = await api.getLayerActivationContexts(selectedLayer);
        console.log(result);
        exampleSets = result.component_example_sets;
        console.log(`loaded ${exampleSets.length} contexts`);
        loading = false;
    }

    onMount(() => {
        loadContexts();
    });
</script>

<div class="activation-contexts-tab">
    <div class="controls">
        <div class="control-group">
            <label for="layer-select">Layer:</label>
            <select id="layer-select" bind:value={selectedLayer}>
                {#each availableComponentLayers as layer}
                    <option value={layer}>{layer}</option>
                {/each}
            </select>
        </div>

        <button class="load-button" on:click={loadContexts}>Load Contexts</button>
    </div>

    {#if exampleSets}
        <div class="components-list">
            {#each exampleSets as exampleSet}
                <div class="component-section">
                    <ActivationContexts
                        component_idx={exampleSet.component_idx}
                        examples={exampleSet.examples}
                    />
                </div>
            {/each}
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

    .control-group select {
        padding: 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 0.9rem;
        min-width: 150px;
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

    .load-button {
        padding: 0.5rem 1rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
    }

    .load-button:hover {
        background: #f8f9fa;
        cursor: pointer;
        background: #007bff;
        color: white;
        border: 1px solid #007bff;
    }
</style>
