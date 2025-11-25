<script lang="ts">
    import type { ComponentDetail, HarvestMetadata } from "../lib/api";
    import * as api from "../lib/api";
    import { logTiming } from "../lib/timing";
    import ActivationContextsPagedTable from "./ActivationContextsPagedTable.svelte";

    interface Props {
        harvestMetadata: HarvestMetadata;
    }

    let { harvestMetadata }: Props = $props();
    if (Object.keys(harvestMetadata.layers).length === 0) {
        throw new Error("No layers data");
    }
    const LIMIT = 20;

    let currentPage = $state(0);
    let selectedLayer = $derived(Object.keys(harvestMetadata.layers)[0]);
    let metricMode = $state<"recall" | "precision">("recall");

    // Display page (1-indexed)
    let displayPage = $derived(currentPage + 1);

    // Update currentPage when page input changes
    function handlePageInput(event: Event) {
        const target = event.target as HTMLInputElement;
        const value = parseInt(target.value);
        if (!isNaN(value) && value >= 1 && value <= totalPages) {
            currentPage = value - 1;
        }
        // If invalid, the derived displayPage will show the correct value
    }

    // Component data cache: key is `${layer}:${componentIdx}`
    let componentCache = $state<Record<string, ComponentDetail>>({});
    let loadingComponent = $state(false);

    // Derive available layers from the data
    let availableComponentLayers = $derived(Object.keys(harvestMetadata.layers));

    // Derive current metadata from selections
    let currentLayerMetadata = $derived(harvestMetadata.layers[selectedLayer]);
    let totalPages = $derived(currentLayerMetadata.length);

    // current page isn't reset to 0 instantly when layer changes, so we must handle the case when,
    // for a brief moment, the current page is out of bounds.
    let currentMetadata = $derived<api.SubcomponentMetadata | null>(currentLayerMetadata.at(currentPage) ?? null);

    // Get current component data from cache
    let currentItem = $derived.by(() => {
        if (!currentMetadata) return null;
        const cacheKey = `${selectedLayer}:${currentMetadata.subcomponent_idx}`;
        return componentCache[cacheKey] ?? null;
    });

    // Debug: inspect when currentItem changes
    $inspect("Viewer", {
        selectedLayer,
        currentPage,
        hasCurrentItem: !!currentItem,
        nExamples: currentItem?.example_tokens.length ?? 0,
    });

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset page when layer changes
    $effect(() => {
        if (selectedLayer) currentPage = 0;
    });

    // Lazy-load component data when page or layer changes
    $effect(() => {
        // establish dependencies by reading them
        const meta = currentMetadata;
        const layer = selectedLayer;

        if (!meta) return;

        const cacheKey = `${layer}:${meta.subcomponent_idx}`;

        // skip if already cached
        if (componentCache[cacheKey]) return;

        let cancelled = false;
        loadingComponent = true;

        const load = async () => {
            const loadStart = performance.now();
            try {
                const detail = await api.getComponentDetail(harvestMetadata.harvest_id, layer, meta.subcomponent_idx);

                if (cancelled) return;
                const updateStart = performance.now();
                componentCache[cacheKey] = detail; // writes to $state
                logTiming("fe_cache_update", performance.now() - updateStart);
                logTiming("fe_load_component_total", performance.now() - loadStart);
            } catch (error) {
                if (!cancelled) console.error("Failed to load component:", error);
            } finally {
                if (!cancelled) loadingComponent = false;
            }
        };

        load();

        // cleanup if deps change before the async work finishes
        return () => {
            cancelled = true;
        };
    });

    // Build sorted token densities from columnar data
    let densities = $derived.by(() => {
        if (!currentItem) return null;
        const n = currentItem.pr_tokens.length;
        // Build array of indices and sort by metric
        const indices = Array.from({ length: n }, (_, i) => i);
        indices.sort((a, b) => {
            const valA = metricMode === "recall" ? currentItem.pr_recalls[a] : currentItem.pr_precisions[a];
            const valB = metricMode === "recall" ? currentItem.pr_recalls[b] : currentItem.pr_precisions[b];
            return valB - valA;
        });
        return indices.slice(0, LIMIT).map((i) => ({
            token: currentItem.pr_tokens[i],
            recall: currentItem.pr_recalls[i],
            precision: currentItem.pr_precisions[i],
        }));
    });
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
    <input type="number" min="1" max={totalPages} value={displayPage} oninput={handlePageInput} class="page-input" />
    <span>of {totalPages}</span>
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
                        {currentItem.pr_tokens.length > LIMIT
                            ? `(top ${LIMIT} of ${currentItem.pr_tokens.length})`
                            : ""}
                    </h5>
                    <div class="metric-toggle">
                        <div class="toggle-buttons">
                            <button class:active={metricMode === "recall"} onclick={() => (metricMode = "recall")}>
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

        <ActivationContextsPagedTable
            exampleTokens={currentItem.example_tokens}
            exampleCi={currentItem.example_ci}
            exampleActivePos={currentItem.example_active_pos}
            activatingTokens={currentItem.pr_tokens}
        />
    </div>
{/if}

<style>
    .layer-select-section {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
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
        padding: 0.5rem 0.5rem;
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
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }

    .pagination-controls button {
        padding: 0.25rem 0.75rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .pagination-controls button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .pagination-controls span {
        font-size: 0.9rem;
        color: #495057;
        white-space: nowrap;
    }

    .page-input {
        width: 60px;
        padding: 0.25rem 0.5rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        text-align: center;
        font-size: 0.9rem;
        appearance: textfield;
    }

    .page-input::-webkit-inner-spin-button,
    .page-input::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    .subcomponent-section-header {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }

    .subcomponent-section-header h4 {
        margin: 0;
        font-size: 1rem;
        color: #495057;
    }

    .token-densities {
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .token-densities-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.5rem;
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
