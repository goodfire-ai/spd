<script lang="ts">
    import type { SubcomponentActivationContexts } from "$lib/api";
    import ActivationContext from "./ActivationContext.svelte";

    export let allLayersData: Record<string, SubcomponentActivationContexts[]>;
    if (Object.keys(allLayersData).length === 0) {
        throw new Error("No layers data");
    }
    const LIMIT = 20;

    let currentPage = 0;
    let selectedLayer: string = Object.keys(allLayersData)[0];
    let metricMode: "recall" | "precision" = "recall";

    // reset selectedLayer to first layer when allLayersData changes
    $: {
        selectedLayer = Object.keys(allLayersData)[0];
    }

    // Derive available layers from the data
    $: availableComponentLayers = Object.keys(allLayersData);

    // Derive current data from selections
    $: currentLayerData = selectedLayer ? allLayersData[selectedLayer] : null;
    $: totalPages = currentLayerData?.length ?? 0;
    $: currentItem = currentLayerData?.[currentPage];

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset page when layer changes
    $: if (selectedLayer) currentPage = 0;

    $: densities = currentItem?.token_densities
        ?.toSorted((a, b) => b[metricMode] - a[metricMode])
        .slice(0, LIMIT);
</script>

<div class="layer-select-section">
    <label for="layer-select">Layer:</label>
    <select id="layer-select" bind:value={selectedLayer}>
        {#each availableComponentLayers as layer (layer)}
            <option value={layer}>{layer}</option>
        {/each}
    </select>

    <div class="metric-toggle">
        <label>Token Metric:</label>
        <div class="toggle-buttons">
            <button class:active={metricMode === "recall"} on:click={() => (metricMode = "recall")}>
                Recall
                <span class="math-notation"> P(token | firing) </span>
            </button>
            <button
                class:active={metricMode === "precision"}
                on:click={() => (metricMode = "precision")}
            >
                Precision
                <span class="math-notation"> P(firing | token) </span>
            </button>
        </div>
    </div>
</div>

<div class="pagination-controls">
    <button on:click={previousPage} disabled={currentPage === 0}>&lt;</button>
    <input type="number" min="0" max={totalPages - 1} bind:value={currentPage} class="page-input" />
    <span>of {totalPages - 1}</span>
    <button on:click={nextPage} disabled={currentPage === totalPages - 1}>&gt;</button>
</div>

{#if currentItem}
    <div class="subcomponent-section-header">
        <h4>
            Subcomponent {currentItem.subcomponent_idx} (Mean CI: {currentItem.mean_ci < 0.001
                ? currentItem.mean_ci.toExponential(2)
                : currentItem.mean_ci.toFixed(3)})
        </h4>
        {#if densities != null}
            <div class="token-densities">
                <h5>
                    Token {metricMode === "recall" ? "Recall" : "Precision"}
                    {currentItem.token_densities.length > LIMIT
                        ? `(top ${LIMIT} of ${currentItem.token_densities.length})`
                        : ""}
                </h5>
                <div class="densities-grid">
                    {#each densities as { token, recall, precision } (`${token}-${recall}-${precision}`)}
                        {@const value = metricMode === "recall" ? recall : precision}
                        <div class="density-item">
                            <span class="token">{token}</span>
                            <div class="density-bar-container">
                                <div class="density-bar" style="width: {value * 100}%"></div>
                            </div>
                            <span class="density-value">{(value * 100).toFixed(1)}%</span>
                        </div>
                    {/each}
                </div>
            </div>
        {/if}

        <div class="subcomponent-section">
            {currentItem.examples.length > 200
                ? `Showing top 200 examples of ${currentItem.examples.length} examples`
                : ""}
            {#each currentItem.examples.slice(0, 200) as example (example.__id)}
                <ActivationContext {example} />
            {/each}
        </div>
    </div>
{/if}

<style>
    .layer-select-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        flex-wrap: wrap;
    }

    .layer-select-section label {
        font-size: 0.9rem;
        color: #495057;
        font-weight: 500;
    }

    #layer-select {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 0.5rem;
        font-size: 0.9rem;
        background: white;
        cursor: pointer;
        min-width: 200px;
    }

    .metric-toggle {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .toggle-buttons {
        display: flex;
        gap: 0;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        overflow: hidden;
    }

    .toggle-buttons button {
        padding: 0.5rem 1rem;
        border: none;
        background: white;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s;
        border-right: 1px solid #dee2e6;
    }

    .toggle-buttons button:last-child {
        border-right: none;
    }

    .toggle-buttons button:hover {
        background: #f8f9fa;
    }

    .toggle-buttons button.active {
        background: #0d6efd;
        color: white;
        font-weight: 500;
    }

    .pagination-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .pagination-controls button {
        padding: 0.5rem 1rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 1rem;
    }

    .pagination-controls button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .page-input {
        width: 60px;
        padding: 0.5rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        text-align: center;
    }

    .subcomponent-section-header {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }

    .subcomponent-section {
        border-radius: 8px;
        overflow: visible;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .token-densities {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .token-densities h5 {
        margin: 0 0 1rem 0;
        font-size: 1rem;
        color: #495057;
    }

    .math-notation {
        font-family: "Georgia", "Times New Roman", serif;
        font-style: italic;
        font-size: 0.9em;
    }

    .densities-grid {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .density-item {
        display: grid;
        grid-template-columns: 100px 1fr 60px;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
    }

    .token {
        font-family: monospace;
        font-weight: 600;
        color: #212529;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .density-bar-container {
        height: 20px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }

    .density-bar {
        height: 100%;
        background: linear-gradient(90deg, #4dabf7, #228be6);
        transition: width 0.3s ease;
    }

    .density-value {
        text-align: right;
        color: #495057;
        font-weight: 500;
    }
</style>
