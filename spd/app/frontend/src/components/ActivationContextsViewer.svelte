<script lang="ts">
    import { run } from 'svelte/legacy';

    import * as api from "../lib/api";
    import type { HarvestMetadata, TokenPR, ComponentDetail } from "../lib/api";
    import ActivationContext from "./ActivationContext.svelte";

    interface Props {
        harvestMetadata: HarvestMetadata;
    }

    let { harvestMetadata }: Props = $props();
    if (Object.keys(harvestMetadata.layers).length === 0) {
        throw new Error("No layers data");
    }
    const LIMIT = 20;

    let currentPage = $state(0);
    let selectedLayer: string = $state(Object.keys(harvestMetadata.layers)[0]);
    let metricMode: "recall" | "precision" = $state("recall");

    // Component data cache: key is `${layer}:${componentIdx}`
    let componentCache = $state<Record<string, ComponentDetail>>({});
    let loadingComponent = $state(false);

    // reset selectedLayer to first layer when harvestMetadata changes
    run(() => {
        selectedLayer = Object.keys(harvestMetadata.layers)[0];
    });

    // Derive available layers from the data
    let availableComponentLayers = $derived(Object.keys(harvestMetadata.layers));

    // Derive current metadata from selections
    let currentLayerMetadata = $derived(selectedLayer ? harvestMetadata.layers[selectedLayer] : null);
    let totalPages = $derived(currentLayerMetadata?.length ?? 0);
    let currentMetadata = $derived(currentLayerMetadata?.[currentPage]);

    // Get current component data from cache
    let currentItem = $derived.by(() => {
        if (!currentMetadata) return null;
        const cacheKey = `${selectedLayer}:${currentMetadata.subcomponent_idx}`;
        return componentCache[cacheKey] ?? null;
    });

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset page when layer changes
    run(() => {
        if (selectedLayer) currentPage = 0;
    });

    // Lazy-load component data when page or layer changes
    run(async () => {
        if (!currentMetadata) return;

        const cacheKey = `${selectedLayer}:${currentMetadata.subcomponent_idx}`;

        // Skip if already cached
        if (componentCache[cacheKey]) return;

        loadingComponent = true;
        try {
            const detail = await api.getComponentDetail(
                harvestMetadata.harvest_id,
                selectedLayer,
                currentMetadata.subcomponent_idx
            );

            // Add to cache
            componentCache[cacheKey] = detail;
        } catch (error) {
            console.error("Failed to load component:", error);
        } finally {
            loadingComponent = false;
        }
    });

    let densities = $derived(currentItem?.token_prs
        ?.slice()
        .sort((a: TokenPR, b: TokenPR) => b[metricMode] - a[metricMode])
        .slice(0, LIMIT));
</script>

<div class="layer-select-section">
    <label for="layer-select">Layer:</label>
    <select id="layer-select" bind:value={selectedLayer}>
        {#each availableComponentLayers as layer (layer)}
            <option value={layer}>{layer}</option>
        {/each}
    </select>
</div>

<div class="pagination-controls">
    <button onclick={previousPage} disabled={currentPage === 0}>&lt;</button>
    <input type="number" min="0" max={totalPages - 1} bind:value={currentPage} class="page-input" />
    <span>of {totalPages - 1}</span>
    <button onclick={nextPage} disabled={currentPage === totalPages - 1}>&gt;</button>
</div>

{#if loadingComponent}
    <div class="loading">Loading component data...</div>
{:else if currentItem && currentMetadata}
    <div class="subcomponent-section-header">
        <h4>
            Subcomponent {currentMetadata.subcomponent_idx} (Mean CI: {currentMetadata.mean_ci < 0.001
                ? currentMetadata.mean_ci.toExponential(2)
                : currentMetadata.mean_ci.toFixed(3)})
        </h4>
        {#if densities != null}
            <div class="token-densities">
                <div class="token-densities-header">
                    <h5>
                        Tokens
                        {currentItem.token_prs.length > LIMIT
                            ? `(top ${LIMIT} of ${currentItem.token_prs.length})`
                            : ""}
                    </h5>
                    <div class="metric-toggle">
                        <div class="toggle-buttons">
                            <button
                                class:active={metricMode === "recall"}
                                onclick={() => (metricMode = "recall")}
                            >
                                Recall
                                <span class="math-notation">P(token | firing)</span>
                            </button>
                            <button
                                class:active={metricMode === "precision"}
                                onclick={() => (metricMode = "precision")}
                            >
                                Precision
                                <span class="math-notation">P(firing | token)</span>
                            </button>
                        </div>
                    </div>
                </div>
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

    .toggle-buttons {
        display: flex;
        gap: 0;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
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

    .token-densities-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .token-densities h5 {
        margin: 0;
        font-size: 1rem;
        color: #495057;
    }

    .math-notation {
        font-family: "Georgia", "Times New Roman", serif;
        font-style: italic;
        font-size: 0.85em;
        margin-left: 0.25rem;
        opacity: 0.8;
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
        background: #4dabf7;
        transition: width 0.3s ease;
    }

    .density-value {
        text-align: right;
        color: #495057;
        font-weight: 500;
    }

    .loading {
        padding: 2rem;
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
    }
</style>
