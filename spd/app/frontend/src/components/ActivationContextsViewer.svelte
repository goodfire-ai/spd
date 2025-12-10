<script lang="ts">
    import type { ComponentDetail, HarvestMetadata } from "../lib/api";
    import type { ComponentCorrelations } from "../lib/localAttributionsTypes";
    import * as api from "../lib/api";
    import { getComponentCorrelations } from "../lib/localAttributionsApi";
    import ActivationContextsPagedTable from "./ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "./ComponentProbeInput.svelte";
    import CorrelationTable from "./local-attr/CorrelationTable.svelte";

    interface Props {
        harvestMetadata: HarvestMetadata;
    }

    let { harvestMetadata }: Props = $props();

    const N_TOKENS_TO_DISPLAY_PR = 100;
    const N_TOKENS_TO_DIPLAY_CORRELATIONS = 20;

    let availableLayers = $derived(Object.keys(harvestMetadata.layers));
    let currentPage = $state(0);
    let selectedLayer = $state<string>(Object.keys(harvestMetadata.layers)[0]);

    let componentCache = $state<Record<string, ComponentDetail>>({});
    let loadingComponent = $state(false);

    // Correlations state
    let correlations = $state<ComponentCorrelations | null>(null);
    let correlationsLoading = $state(false);

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

        return () => {
            cancelled = true;
        };
    });

    // Correlation job status (for showing message when no data)
    let correlationJobStatus = $state<api.CorrelationJobStatus | null>(null);

    // Fetch correlation job status once on mount
    $effect(() => {
        api.getCorrelationJobStatus().then((s) => {
            correlationJobStatus = s;
        });
    });

    // Fetch correlations when component changes
    $effect(() => {
        const layer = selectedLayer;
        const cIdx = currentMetadata?.subcomponent_idx;
        if (cIdx === undefined) return;

        correlations = null;
        correlationsLoading = true;
        getComponentCorrelations(layer, cIdx, N_TOKENS_TO_DIPLAY_CORRELATIONS)
            .then((data) => {
                correlations = data;
            })
            .catch(() => {
                correlations = null;
            })
            .finally(() => {
                correlationsLoading = false;
            });
    });

    // Tokens by recall (already sorted by backend)
    let topRecall = $derived.by(() => {
        if (!currentComponent) return [];
        return currentComponent.top_recall.slice(0, N_TOKENS_TO_DISPLAY_PR).map(([token, value]) => ({
            token,
            value,
        }));
    });

    // Tokens by precision (already sorted by backend)
    let topPrecision = $derived.by(() => {
        if (!currentComponent) return [];
        return currentComponent.top_precision.slice(0, N_TOKENS_TO_DISPLAY_PR).map(([token, value]) => ({
            token,
            value,
        }));
    });

    // Background color for token pills (green intensity)
    function getTokenBg(value: number): string {
        return `rgba(22, 163, 74, ${value})`;
    }
</script>

<div class="viewer-content">
    <div class="controls-row">
        <div class="layer-select">
            <label for="layer-select">Layer:</label>
            <select id="layer-select" bind:value={selectedLayer}>
                {#each availableLayers as layer (layer)}
                    <option value={layer}>{layer}</option>
                {/each}
            </select>
        </div>

        <div class="pagination">
            <label for="page-input">Subcomponent:</label>
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
    </div>

    {#if loadingComponent}
        <div class="loading">Loading component data...</div>
    {:else if currentComponent && currentMetadata}
        <div class="component-section">
            <h4>
                Subcomponent {currentMetadata.subcomponent_idx}
                <span class="mean-ci">Mean CI: {currentMetadata.mean_ci < 0.001
                    ? currentMetadata.mean_ci.toExponential(2)
                    : currentMetadata.mean_ci.toFixed(3)}</span>
            </h4>

            <!-- Token statistics: Recall and Precision -->
            {#if topRecall.length > 0}
                <div class="token-stats">
                    <div class="token-list">
                        <h5>Recall <span class="math-notation">P(token | firing)</span></h5>
                        <div class="tokens">
                            {#each topRecall as { token, value } (token)}
                                <span class="token-pill" style="background: {getTokenBg(value)}" title="{(value * 100).toFixed(1)}%">{token}</span>
                            {/each}
                        </div>
                    </div>
                    <div class="token-list">
                        <h5>Precision <span class="math-notation">P(firing | token)</span></h5>
                        <div class="tokens">
                            {#each topPrecision as { token, value } (token)}
                                <span class="token-pill" style="background: {getTokenBg(value)}" title="{(value * 100).toFixed(1)}%">{token}</span>
                            {/each}
                        </div>
                    </div>
                </div>
            {/if}

            <ComponentProbeInput layer={selectedLayer} componentIdx={currentMetadata.subcomponent_idx} />

            <!-- Component correlations -->
            <div class="correlations-section">
                <h5>Correlated Components</h5>
                {#if correlations}
                    <div class="correlations-grid">
                        <CorrelationTable
                            title="Precision"
                            mathNotation="P(that | this)"
                            items={correlations.precision}
                            maxItems={N_TOKENS_TO_DIPLAY_CORRELATIONS}
                        />
                        <CorrelationTable
                            title="Recall"
                            mathNotation="P(this | that)"
                            items={correlations.recall}
                            maxItems={N_TOKENS_TO_DIPLAY_CORRELATIONS}
                        />
                        <CorrelationTable
                            title="F1"
                            items={correlations.f1}
                            maxItems={N_TOKENS_TO_DIPLAY_CORRELATIONS}
                        />
                        <CorrelationTable
                            title="Jaccard"
                            items={correlations.jaccard}
                            maxItems={N_TOKENS_TO_DIPLAY_CORRELATIONS}
                        />
                    </div>
                {:else if correlationsLoading}
                    <p class="status-text muted">Loading...</p>
                {:else if correlationJobStatus === null}
                    <p class="status-text muted">No correlations data. Use "Harvest" in the top bar to compute.</p>
                {:else if correlationJobStatus.status === "pending"}
                    <p class="status-text pending">Job {correlationJobStatus.job_id} pending...</p>
                {:else if correlationJobStatus.status === "running"}
                    <p class="status-text running">Job {correlationJobStatus.job_id} running...</p>
                {:else if correlationJobStatus.status === "failed"}
                    <p class="status-text failed">Correlation job failed. Check top bar to retry.</p>
                {:else}
                    <p class="status-text muted">No correlations for this component.</p>
                {/if}
            </div>

            <ActivationContextsPagedTable
                exampleTokens={currentComponent.example_tokens}
                exampleCi={currentComponent.example_ci}
                exampleActivePos={currentComponent.example_active_pos}
                activatingTokens={currentComponent.top_recall.map(([t]) => t)}
            />
        </div>
    {/if}
</div>

<style>
    .viewer-content {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .controls-row {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        flex-wrap: wrap;
    }

    .layer-select {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .viewer-content label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    #layer-select {
        border: 1px solid var(--border-default);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 180px;
    }

    #layer-select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .pagination button {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-secondary);
    }

    .pagination button:hover:not(:disabled) {
        background: var(--bg-surface);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .pagination button:disabled {
        opacity: 0.5;
    }

    .pagination span {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .page-input {
        width: 50px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        appearance: textfield;
    }

    .page-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .page-input::-webkit-inner-spin-button,
    .page-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .component-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .component-section h4 {
        margin: 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .mean-ci {
        font-weight: 400;
        color: var(--text-muted);
        font-family: var(--font-mono);
        margin-left: var(--space-2);
    }

    .token-stats {
        display: flex;
        flex-direction: column;
    }

    .token-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .token-list h5 {
        margin: 0;
        font-size: var(--text-xs);
        font-family: var(--font-sans);
        color: var(--text-muted);
        font-weight: 500;
    }

    .math-notation {
        font-family: var(--font-mono);
        font-style: normal;
        font-size: var(--text-xs);
        opacity: 0.7;
    }

    .tokens {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-1);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
    }

    .token-pill {
        padding: 1px 4px;
        border-radius: 3px;
        white-space: pre;
        cursor: default;
        position: relative;
    }

    .token-pill::after {
        content: attr(title);
        position: absolute;
        bottom: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: 2px 6px;
        font-size: var(--text-xs);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 100;
        border-radius: 3px;
    }

    .token-pill:hover::after {
        opacity: 1;
    }

    .status-text {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        margin: 0;
    }

    .status-text.muted {
        color: var(--text-muted);
    }

    .status-text.pending,
    .status-text.running {
        color: var(--accent-primary);
    }

    .status-text.failed {
        color: var(--status-negative-bright);
    }

    .correlations-section {
        margin-top: var(--space-2);
    }

    .correlations-section h5 {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .correlations-grid {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-3);
    }

    .loading {
        padding: var(--space-4);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
