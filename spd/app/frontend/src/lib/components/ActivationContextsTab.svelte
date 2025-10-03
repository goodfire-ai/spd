<script lang="ts">
    import type { ComponentActivationContexts } from "$lib/api";
    import * as api from "$lib/api";
    import { onMount } from "svelte";
    import ActivationContext from "./ActivationContext.svelte";

    export let availableComponentLayers: string[];

    let selectedLayer: string = availableComponentLayers[0];

    let exampleSets: ComponentActivationContexts[] | null = null;

    let loading = false;

    let currentAbort: AbortController | null = null;

    async function loadContexts() {
        if (currentAbort) {
            currentAbort.abort();
        }
        const ac = new AbortController();
        currentAbort = ac;
        loading = true;
        try {
            console.log(`loading contexts for layer ${selectedLayer}`);
            exampleSets = await api.getLayerActivationContexts(selectedLayer, ac.signal);
            console.log(`loaded ${exampleSets.length} contexts`);
        } catch (e) {
            if ((e as any)?.name !== "AbortError") {
                console.error(e);
            }
        } finally {
            if (currentAbort === ac) {
                currentAbort = null;
            }
            loading = false;
        }
    }

    onMount(() => {
        loadContexts();
    });
</script>

<div class="tab-content">
    <div class="controls">
        <div class="control-group">
            <label for="layer-select">Layer:</label>
            <select id="layer-select" bind:value={selectedLayer} on:change={loadContexts}>
                {#each availableComponentLayers as layer}
                    <option value={layer}>{layer}</option>
                {/each}
            </select>
        </div>

        <button class="load-button" on:click={loadContexts}>Load Contexts</button>
    </div>

    {#if exampleSets}
        <div class="subcomponents-list">
            {#each exampleSets as exampleSet}
                <div class="subcomponent-section-header">
                    <h4>Subcomponent {exampleSet.subcomponent_idx}</h4>
                    <div class="subcomponent-section">
                        {#each exampleSet.examples as example}
                            <ActivationContext {example} />
                        {/each}
                    </div>
                </div>
            {/each}
        </div>
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

    .subcomponent-section-header {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }

    .subcomponents-list {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        overflow-y: auto;
        padding: 0.5rem;
    }

    .subcomponent-section {
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
