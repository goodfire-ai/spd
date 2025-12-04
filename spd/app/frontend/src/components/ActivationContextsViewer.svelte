<script lang="ts">
    import type { ComponentDetail, HarvestMetadata } from "../lib/api";
    import * as api from "../lib/api";
    import ActivationContextsPagedTable from "./ActivationContextsPagedTable.svelte";

    interface Props {
        harvestMetadata: HarvestMetadata;
    }

    let { harvestMetadata }: Props = $props();

    const N_TOKENS_TO_DISPLAY = 20;

    let availableLayers = $derived(Object.keys(harvestMetadata.layers));
    let currentPage = $state(0);
    let selectedLayer = $state<string>(Object.keys(harvestMetadata.layers)[0]);
    let metricMode = $state<"recall" | "precision">("recall");

    let componentCache = $state<Record<string, ComponentDetail>>({});
    let loadingComponent = $state(false);

    let currentLayerMetadata = $derived(harvestMetadata.layers[selectedLayer]);
    let totalPages = $derived(currentLayerMetadata.length);
    let currentMetadata = $derived<api.SubcomponentMetadata>(currentLayerMetadata[currentPage]);

    function getCacheKey(layer: string, componentIdx: number) {
        return `${layer}:${componentIdx}`;
    }

    let currentComponent = $derived.by(() => {
        const cacheKey = getCacheKey(selectedLayer, currentMetadata.subcomponent_idx);
        return componentCache[cacheKey];
    });

    function handlePageInput(event: Event) {
        const target = event.target as HTMLInputElement;
        if (target.value === "") return;
        const value = parseInt(target.value);
        if (!isNaN(value) && value >= 1 && value <= totalPages) {
            currentPage = value - 1;
        }
    }

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset page when layer changes
    $effect(() => {
        selectedLayer; // eslint-disable-line @typescript-eslint/no-unused-expressions
        currentPage = 0;
    });

    // Lazy-load component data when page or layer changes
    $effect(() => {
        // establish dependencies by reading them
        const meta = currentMetadata;
        const layer = selectedLayer;

        if (!meta) return;

        const cacheKey = getCacheKey(layer, meta.subcomponent_idx);

        // skip if already cached
        if (componentCache[cacheKey]) return;

        let cancelled = false;
        loadingComponent = true;

        const load = async () => {
            try {
                const detail = await api.getComponentDetail(layer, meta.subcomponent_idx);

                if (cancelled) return;
                componentCache[cacheKey] = detail;
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
        const n = currentComponent.pr_tokens.length;
        const indices = Array.from({ length: n }, (_, i) => i);
        indices.sort((a, b) => {
            const valA = metricMode === "recall" ? currentComponent.pr_recalls[a] : currentComponent.pr_precisions[a];
            const valB = metricMode === "recall" ? currentComponent.pr_recalls[b] : currentComponent.pr_precisions[b];
            return valB - valA;
        });
        return indices.slice(0, N_TOKENS_TO_DISPLAY).map((i) => ({
            token: currentComponent.pr_tokens[i],
            recall: currentComponent.pr_recalls[i],
            precision: currentComponent.pr_precisions[i],
        }));
    });
</script>

<div class="layer-select-section">
    <label for="layer-select">Layer:</label>
    <select id="layer-select" bind:value={selectedLayer}>
        {#each availableLayers as layer (layer)}
            <option value={layer}>{layer}</option>
        {/each}
    </select>
</div>

<div class="pagination-controls">
    <button onclick={previousPage} disabled={currentPage === 0}>&lt;</button>
    <input
        type="number"
        min="1"
        max={totalPages}
        value={currentPage + 1}
        oninput={handlePageInput}
        class="page-input"
    />
    <span>of {totalPages}</span>
    <button onclick={nextPage} disabled={currentPage === totalPages - 1}>&gt;</button>
</div>

{#if loadingComponent}
    <div class="loading">Loading component data...</div>
{:else if currentComponent && currentMetadata}
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
                        {currentComponent.pr_tokens.length > N_TOKENS_TO_DISPLAY
                            ? `(top ${N_TOKENS_TO_DISPLAY} of ${currentComponent.pr_tokens.length})`
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
            exampleTokens={currentComponent.example_tokens}
            exampleCi={currentComponent.example_ci}
            exampleActivePos={currentComponent.example_active_pos}
            activatingTokens={currentComponent.pr_tokens}
        />
    </div>
{/if}

<style>
    .layer-select-section {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        flex-wrap: wrap;
    }

    .layer-select-section label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    #layer-select {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 200px;
    }

    #layer-select:focus {
        outline: none;
        border-color: var(--accent-warm-dim);
    }

    .toggle-buttons {
        display: flex;
        gap: 0;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        overflow: hidden;
    }

    .toggle-buttons button {
        padding: var(--space-1) var(--space-2);
        border: none;
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        border-right: 1px solid var(--border-default);
    }

    .toggle-buttons button:last-child {
        border-right: none;
    }

    .toggle-buttons button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .toggle-buttons button.active {
        background: var(--accent-warm);
        color: white;
        font-weight: 500;
    }

    .pagination-controls {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
    }

    .pagination-controls button {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
    }

    .pagination-controls button:hover:not(:disabled) {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .pagination-controls button:disabled {
        opacity: 0.3;
        cursor: not-allowed;
    }

    .pagination-controls span {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .page-input {
        width: 50px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        appearance: textfield;
    }

    .page-input:focus {
        outline: none;
        border-color: var(--accent-warm-dim);
    }

    .page-input::-webkit-inner-spin-button,
    .page-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .subcomponent-section-header {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .subcomponent-section-header h4 {
        margin: 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .token-densities {
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
    }

    .token-densities-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
        gap: var(--space-2);
        flex-wrap: wrap;
    }

    .token-densities h5 {
        margin: 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .math-notation {
        font-family: var(--font-mono);
        font-style: normal;
        font-size: var(--text-xs);
        margin-left: var(--space-1);
        opacity: 0.7;
    }

    .densities-grid {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .density-item {
        display: grid;
        grid-template-columns: 100px 1fr 60px;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-sm);
    }

    .token {
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--text-primary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: pre;
    }

    .density-bar-container {
        height: 4px;
        background: var(--border-default);
        border-radius: 2px;
        overflow: hidden;
    }

    .density-bar {
        height: 100%;
        background: var(--status-info);
        border-radius: 2px;
        transition: width 0.15s ease-out;
    }

    .density-value {
        text-align: right;
        font-family: var(--font-mono);
        color: var(--text-muted);
        font-weight: 500;
    }

    .loading {
        padding: var(--space-4);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
