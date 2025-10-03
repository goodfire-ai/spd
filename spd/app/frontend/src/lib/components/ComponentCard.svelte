<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type {
        ActivationContext as ActivationContextType,
        Component,
        CosineSimilarityData
    } from "$lib/api";
    import { getCosineSimilarities, getSubcomponentActivationContexts } from "$lib/api";
    import { ablationComponentMask } from "$lib/stores/componentState";
    import { onMount } from "svelte";
    import CosineSimilarityPlot from "./CosineSimilarityPlot.svelte";
    import ActivationContext from "./ActivationContext.svelte";

    export let component: Component;
    export let componentAggCi: number;
    export let subcomponentCis: number[];
    export let layer: string;
    export let tokenIdx: number;
    export let onToggle: () => void;

    let similarityData: CosineSimilarityData | null = null;
    let loadingSimilarities = false;

    $: isDisabled = (() => {
        const layerMask = $ablationComponentMask[layer];
        if (!layerMask || !layerMask[tokenIdx]) return false;
        return layerMask[tokenIdx].includes(component.index);
    })();

    function getColorFromCI(ci: number): string {
        const whiteAmount = Math.round((1 - ci) * 255);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    }

    async function loadCosineSimilarities() {
        if (similarityData) return;

        loadingSimilarities = true;
        try {
            similarityData = await getCosineSimilarities(layer, component.index);
        } catch (error) {
            console.error("Failed to load cosine similarities:", error);
            similarityData = null;
        }
        loadingSimilarities = false;
    }

    let loadingActivationContexts = false;
    let examples: ActivationContextType[] = [];

    async function loadActivationContexts() {
        loadingActivationContexts = true;
        // examples = await getComponentActivationContexts(component.index, layer);
        loadingActivationContexts = false;
    }

    function handleCardClick() {
        onToggle();
    }

    onMount(() => {
        loadCosineSimilarities();
        loadActivationContexts();
    });

    $: disabledComponentIndices = isDisabled ? [component.index] : [];
</script>

<div class="component-card-container" class:disabled={isDisabled} on:click={handleCardClick}>
    <div
        class="component-header-bar"
        style="background-color: {getColorFromCI(componentAggCi / 2)}"
    >
        <span class="component-index">#{component.index}</span>
        <span class="component-ci">{componentAggCi.toFixed(4)}</span>
        <span class="component-rank">{component.subcomponent_indices.length}</span>
    </div>

    <div class="subcomponent-ci-strip">
        {#each subcomponentCis as ci, idx}
            <div
                class="ci-cell"
                style="background-color: {getColorFromCI(ci / 2)}"
                title="Subcomponent {component.subcomponent_indices[idx]}: CI = {ci.toFixed(4)}"
            ></div>
        {/each}
    </div>

    <div class="card-content">
        <div class="activation-contexts-section">
            <h4>Activation Examples</h4>
            {#if loadingActivationContexts}
                <div class="loading">Loading examples...</div>
            {:else}
                <div class="examples-container">
                    {#each examples as example}
                        <ActivationContext {example} />
                    {/each}
                </div>
            {/if}
        </div>

        {#if loadingSimilarities}
            <div class="loading-similarities">Loading similarity data...</div>
        {:else if similarityData}
            <div class="similarity-plots">
                <h4>Pairwise Cosine Similarities</h4>
                <div class="plots-container">
                    <div class="plot-wrapper">
                        <CosineSimilarityPlot
                            title="Input"
                            data={similarityData.input_singular_vectors}
                            indices={similarityData.component_indices}
                            disabledIndices={disabledComponentIndices}
                        />
                    </div>
                    <div class="plot-wrapper">
                        <CosineSimilarityPlot
                            title="Output"
                            data={similarityData.output_singular_vectors}
                            indices={similarityData.component_indices}
                            disabledIndices={disabledComponentIndices}
                        />
                    </div>
                </div>
            </div>
        {/if}
    </div>
</div>

<style>
    .component-card-container {
        display: flex;
        flex-direction: column;
        background-color: white;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e0e0e0;
        position: relative;
        min-width: 600px;
        max-width: 600px;
        height: 800px;
        overflow: hidden;
        flex-shrink: 0;
    }

    .component-card-container:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border-color: rgba(0, 0, 0, 0.2);
    }

    .component-card-container.disabled .component-header-bar {
        background-color: #ff6b6b !important;
    }

    .component-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        flex-shrink: 0;
    }

    .subcomponent-ci-strip {
        display: flex;
        height: 8px;
        flex-shrink: 0;
        border-bottom: 1px solid #e0e0e0;
    }

    .ci-cell {
        flex: 1;
        min-width: 2px;
        cursor: help;
    }

    .component-index {
        font-weight: 600;
        color: #1a1a1a;
        font-size: 1rem;
    }

    .component-ci {
        font-weight: 700;
        color: #1a1a1a;
        font-size: 0.9rem;
        font-family: "Monaco", "Courier New", monospace;
    }

    .component-rank {
        font-weight: 700;
        color: #1a1a1a;
        font-size: 0.9rem;
        font-family: "Monaco", "Courier New", monospace;
    }

    .card-content {
        display: flex;
        flex-direction: column;
        flex: 1;
        overflow: hidden;
    }

    .activation-contexts-section {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .activation-contexts-section h4 {
        margin: 0 0 0.75rem 0;
        color: #333;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .loading,
    .loading-similarities {
        padding: 1rem;
        text-align: center;
        color: #666;
        font-style: italic;
        font-size: 0.85rem;
    }

    .similarity-plots {
        padding: 1rem;
        background: #f8f9fa;
        flex-shrink: 0;
    }

    .similarity-plots h4 {
        margin: 0 0 0.75rem 0;
        color: #333;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .plots-container {
        display: flex;
        gap: 1rem;
    }

    .plot-wrapper {
        flex: 1;
        min-width: 0;
    }

    .examples-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        overflow-y: auto;
    }
</style>
