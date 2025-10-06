<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type { CosineSimilarityData } from "$lib/api";
    import * as api from "$lib/api";
    import { ablationComponentMask } from "$lib/stores/componentState";
    import { onMount } from "svelte";
    import CosineSimilarityPlot from "./CosineSimilarityPlot.svelte";
    import TokenHighlights from "./TokenHighlights.svelte";

    type ComponentExample = {
        textHash: string;
        rawText: string;
        offsetMapping: [number, number][];
        activations: number[];
    };

    export let layer: string;
    export let tokenIdx: number;
    export let componentIdx: number;
    export let componentAggCi: number;
    export let subcomponentCis: number[];
    export let toggle: () => void;
    export let examples: ComponentExample[] = [];

    $: isDisabled = (() => {
        const layerMask = $ablationComponentMask[layer];
        if (!layerMask || !layerMask[tokenIdx]) return false;
        return layerMask[tokenIdx].includes(componentIdx);
    })();

    function getColorFromCI(ci: number): string {
        const whiteAmount = Math.round((1 - ci) * 255);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    }

    $: disabledComponentIndices = isDisabled ? [componentIdx] : [];

    $: textColor = componentAggCi > 0.5 ? "#ffffff" : "#000000";

    let similarityData: CosineSimilarityData | null = null;
    let loading = false;
    async function loadCosineSims() {
        loading = true;
        await new Promise((resolve) => setTimeout(resolve, componentIdx * 50));
        similarityData = await api.getCosineSimilarities(layer, componentIdx);
        loading = false;
    }
    onMount(loadCosineSims);
</script>

<div class="component-card-container" class:disabled={isDisabled} on:click={toggle}>
    <div class="component-header-bar" style="background-color: {getColorFromCI(componentAggCi)}">
        <span class="component-index" style="color: {textColor}">#{componentIdx}</span>
        <span class="component-ci" style="color: {textColor}">{componentAggCi.toFixed(4)}</span>
        <span class="component-rank" style="color: {textColor}">Rank {subcomponentCis.length}</span>
    </div>

    <div class="subcomponent-ci-strip">
        {#each subcomponentCis as ci, idx}
            <div
                class="ci-cell"
                style="background-color: {getColorFromCI(ci)}"
                title="Subcomponent {idx}: CI = {ci.toFixed(4)}"
            ></div>
        {/each}
    </div>

    <div class="card-content">
        <div class="activation-contexts-section">
            <h4>Activation Examples</h4>
            <div class="examples-container">
                {#each examples as example}
                    <div class="example-card">
                        <TokenHighlights
                            rawText={example.rawText}
                            offsetMapping={example.offsetMapping}
                            tokenCiValues={example.activations}
                            activePosition={-1}
                        />
                    </div>
                {/each}
            </div>
        </div>

        {#if similarityData}
            <div class="similarity-plots">
                <h4>Pairwise Cosine Similarities</h4>
                <div class="plots-container">
                    <div class="plot-wrapper">
                        <h5>Input</h5>
                        <CosineSimilarityPlot
                            data={similarityData.input_singular_vectors}
                            indices={similarityData.component_indices}
                            disabledIndices={disabledComponentIndices}
                        />
                    </div>
                    <div class="plot-wrapper">
                        <h5>Output</h5>
                        <CosineSimilarityPlot
                            data={similarityData.output_singular_vectors}
                            indices={similarityData.component_indices}
                            disabledIndices={disabledComponentIndices}
                        />
                    </div>
                </div>
            </div>
        {:else}
            <div class="loading-similarities">
                <p>Loading...</p>
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
        /* min-width: 600px; */
        width: 600px;
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
        /* cursor: help; */
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

    .example-card {
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 0.5rem;
        background: #fdfdfd;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.5;
    }
</style>
