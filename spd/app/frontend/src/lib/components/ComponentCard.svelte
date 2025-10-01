<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type { Component, CosineSimilarityData } from "$lib/api";
    import { api } from "$lib/api";
    import { ablationComponentMask } from "$lib/stores/componentState";
    import CosineSimilarityPlot from "./CosineSimilarityPlot.svelte";

    export let component: Component;
    export let componentAggCi: number;
    export let layer: string;
    export let tokenIdx: number;
    export let onToggle: () => void;

    let similarityData: CosineSimilarityData | null = null;
    let loadingSimilarities = false;
    let expanded = false;

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
            similarityData = await api.getCosineSimilarities(layer, component.index);
        } catch (error) {
            console.error("Failed to load cosine similarities:", error);
            similarityData = null;
        }
        loadingSimilarities = false;
    }

    function handleCardClick() {
        onToggle();
    }

    function handleExpandClick(e: MouseEvent) {
        e.stopPropagation();
        expanded = !expanded;
        if (expanded && !similarityData) {
            loadCosineSimilarities();
        }
    }

    $: disabledComponentIndices = isDisabled ? [component.index] : [];
</script>

<div class="component-card-container">
    <div
        class="component-card"
        style="--component-bg: {getColorFromCI(componentAggCi / 2)}"
        class:disabled={isDisabled}
        on:click={handleCardClick}
    >
        <div class="component-header">
            <span class="component-index">#{component.index}</span>
            <span class="component-ci">{componentAggCi.toFixed(4)}</span>
            <span class="component-rank">{component.subcomponent_indices.length}</span>
        </div>
        <button
            class="expand-button"
            on:click={handleExpandClick}
            title={expanded ? "Collapse" : "Expand"}
        >
            {expanded ? "▼" : "▶"}
        </button>
    </div>

    {#if expanded}
        <div class="expanded-content">
            {#if loadingSimilarities}
                <div class="loading-similarities">Loading similarity data...</div>
            {:else if similarityData}
                <div class="similarity-plots">
                    <h4>Pairwise Cosine Similarities</h4>
                    <div class="plots-container">
                        <CosineSimilarityPlot
                            title="Input Singular Vectors"
                            data={similarityData.input_singular_vectors}
                            indices={similarityData.component_indices}
                            disabledIndices={disabledComponentIndices}
                        />
                        <CosineSimilarityPlot
                            title="Output Singular Vectors"
                            data={similarityData.output_singular_vectors}
                            indices={similarityData.component_indices}
                            disabledIndices={disabledComponentIndices}
                        />
                    </div>
                </div>
            {/if}
        </div>
    {/if}
</div>

<style>
    .component-card-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .component-card {
        background-color: var(--component-bg);
        border-radius: 6px;
        padding: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 2px solid transparent;
        position: relative;
    }

    .component-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-color: rgba(0, 0, 0, 0.1);
    }

    .component-card.disabled {
        background-color: #ff6b6b !important;
        opacity: 0.8;
    }

    .component-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .component-index {
        font-weight: 600;
        color: #333;
        font-size: 0.95rem;
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

    .expand-button {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        cursor: pointer;
        font-size: 0.8rem;
        transition: background 0.2s;
    }

    .expand-button:hover {
        background: rgba(255, 255, 255, 0.9);
    }

    .expanded-content {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #e9ecef;
    }

    .loading-similarities {
        padding: 1rem;
        text-align: center;
        color: #666;
        font-style: italic;
    }

    .similarity-plots h4 {
        margin: 0 0 1rem 0;
        color: #333;
        font-size: 0.9rem;
    }

    .plots-container {
        display: flex;
        gap: 2rem;
        justify-content: space-around;
        flex-wrap: wrap;
    }
</style>
