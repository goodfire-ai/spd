<script lang="ts">
    import type { Loadable } from "../lib";
    import type { SubcomponentActivationContexts, HarvestMetadata } from "../lib/api";
    import * as api from "../lib/api";
    import { useComponentData } from "../lib/useComponentData.svelte";
    import ActivationContextsPagedTable from "./ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "./ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "./ui/ComponentCorrelationMetrics.svelte";
    import InterpretationBadge from "./ui/InterpretationBadge.svelte";
    import SectionHeader from "./ui/SectionHeader.svelte";
    import StatusText from "./ui/StatusText.svelte";
    import TokenStatsSection from "./ui/TokenStatsSection.svelte";

    interface Props {
        harvestMetadata: HarvestMetadata;
    }

    let { harvestMetadata }: Props = $props();

    const N_TOKENS_TO_DISPLAY_INPUT = 80;
    const N_TOKENS_TO_DISPLAY_OUTPUT = 30;

    let availableLayers = $derived(Object.keys(harvestMetadata.layers).sort());
    let currentPage = $state(0);
    let selectedLayer = $state<string>(Object.keys(harvestMetadata.layers)[0]);

    let componentCache = $state<Record<string, Loadable<SubcomponentActivationContexts>>>({});
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- not reactive, just for deduplication
    const requestedKeys = new Set<string>();

    // Layer metadata is already sorted by mean_ci desc from backend
    let currentLayerMetadata = $derived(harvestMetadata.layers[selectedLayer]);
    let totalPages = $derived(currentLayerMetadata.length);
    let currentMetadata = $derived<api.SubcomponentMetadata>(currentLayerMetadata[currentPage]);

    // Fetch correlations, token stats, and interpretation for current component
    const componentData = useComponentData(() => {
        const cIdx = currentMetadata?.subcomponent_idx;
        return cIdx !== undefined ? { layer: selectedLayer, cIdx } : null;
    });

    function getCacheKey(layer: string, componentIdx: number) {
        return `${layer}:${componentIdx}`;
    }

    let currentComponent = $derived.by(() => {
        const cacheKey = getCacheKey(selectedLayer, currentMetadata.subcomponent_idx);
        return componentCache[cacheKey] ?? null;
    });

    function handlePageInput(event: Event) {
        const target = event.target as HTMLInputElement;
        if (target.value === "") return;
        const value = parseInt(target.value);
        if (!isNaN(value) && value >= 1 && value <= totalPages) {
            currentPage = value - 1;
        }
    }

    // Search for a specific subcomponent index
    let searchValue = $state("");
    let searchError = $state<string | null>(null);

    function handleSearchInput(event: Event) {
        const target = event.target as HTMLInputElement;
        searchValue = target.value;
        searchError = null;

        if (searchValue === "") return;

        const targetIdx = parseInt(searchValue);
        if (isNaN(targetIdx)) {
            searchError = "Invalid number";
            return;
        }

        // Find the page index that contains this subcomponent index
        const pageIndex = currentLayerMetadata.findIndex((m) => m.subcomponent_idx === targetIdx);

        if (pageIndex === -1) {
            searchError = `Not found`;
            return;
        }

        currentPage = pageIndex;
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

        if (requestedKeys.has(cacheKey)) return;
        requestedKeys.add(cacheKey);

        let cancelled = false;
        componentCache[cacheKey] = { status: "loading" };

        const load = async () => {
            try {
                const detail = await api.getComponentDetail(layer, meta.subcomponent_idx);

                if (cancelled) return;
                componentCache[cacheKey] = { status: "loaded", data: detail };
            } catch (error) {
                if (!cancelled) componentCache[cacheKey] = { status: "error", error };
            }
        };

        load();

        return () => {
            cancelled = true;
        };
    });

    // Derive token lists from loaded tokenStats (null if not loaded or no data)
    const inputTokenLists = $derived.by(() => {
        const ts = componentData.tokenStats;
        if (ts?.status !== "loaded" || ts.data === null) return null;
        return [
            {
                title: "Top Recall",
                mathNotation: "P(token | component fires)",
                items: ts.data.input.top_recall
                    .slice(0, N_TOKENS_TO_DISPLAY_INPUT)
                    .map(([token, value]) => ({ token, value })),
            },
            {
                title: "Top Precision",
                mathNotation: "P(component fires | token)",
                items: ts.data.input.top_precision
                    .slice(0, N_TOKENS_TO_DISPLAY_INPUT)
                    .map(([token, value]) => ({ token, value })),
            },
        ];
    });

    const outputTokenLists = $derived.by(() => {
        const ts = componentData.tokenStats;
        if (ts?.status !== "loaded" || ts.data === null) return null;
        return [
            {
                title: "Top PMI",
                mathNotation: "positive association with predictions",
                items: ts.data.output.top_pmi
                    .slice(0, N_TOKENS_TO_DISPLAY_OUTPUT)
                    .map(([token, value]) => ({ token, value })),
            },
            {
                title: "Bottom PMI",
                mathNotation: "negative association with predictions",
                items: ts.data.output.bottom_pmi
                    .slice(0, N_TOKENS_TO_DISPLAY_OUTPUT)
                    .map(([token, value]) => ({ token, value })),
            },
        ];
    });

    // Activating tokens from token stats (for highlighting in table)
    let inputTopRecall = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats?.status !== "loaded" || tokenStats.data === null) return [];
        return tokenStats.data.input.top_recall.map(([token, value]) => ({ token, value }));
    });

    // Format mean CI for display
    function formatMeanCi(ci: number): string {
        return ci < 0.001 ? ci.toExponential(2) : ci.toFixed(3);
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

        <div class="search-box">
            <label for="search-input">Go to index:</label>
            <input
                id="search-input"
                type="number"
                placeholder="e.g. 42"
                value={searchValue}
                oninput={handleSearchInput}
                class="search-input"
            />
            {#if searchError}
                <span class="search-error">{searchError}</span>
            {/if}
        </div>
    </div>

    <div class="component-section">
        <SectionHeader title="Subcomponent {currentMetadata.subcomponent_idx}" level="h4">
            <span class="mean-ci">Mean CI: {formatMeanCi(currentMetadata.mean_ci)}</span>
        </SectionHeader>

        <InterpretationBadge interpretation={componentData.interpretation} onGenerate={componentData.generateInterpretation} />

        <div class="token-stats-row">
            {#if componentData.tokenStats === null || componentData.tokenStats.status === "loading"}
                <StatusText>Loading token stats...</StatusText>
            {:else if componentData.tokenStats.status === "error"}
                <StatusText>Error: {String(componentData.tokenStats.error)}</StatusText>
            {:else}
                <TokenStatsSection
                    sectionTitle="Input Tokens"
                    sectionSubtitle="(what activates this component)"
                    lists={inputTokenLists}
                />

                <TokenStatsSection
                    sectionTitle="Output Tokens"
                    sectionSubtitle="(what this component predicts)"
                    lists={outputTokenLists}
                />
            {/if}
        </div>

        <!-- Component correlations -->
        <div class="correlations-section">
            <SectionHeader title="Correlated Components" />
            {#if componentData.correlations === null || componentData.correlations.status === "loading"}
                <StatusText>Loading...</StatusText>
            {:else if componentData.correlations.status === "error"}
                <StatusText>Error loading correlations: {String(componentData.correlations.error)}</StatusText>
            {:else if componentData.correlations.data === null}
                <StatusText>No correlations data. Run harvest pipeline first.</StatusText>
            {:else}
                <ComponentCorrelationMetrics correlations={componentData.correlations.data} pageSize={40} />
            {/if}
        </div>

        <ComponentProbeInput layer={selectedLayer} componentIdx={currentMetadata.subcomponent_idx} />

        {#if currentComponent?.status === "loading"}
            <div class="loading">Loading component data...</div>
        {:else if currentComponent?.status === "loaded"}
            <ActivationContextsPagedTable
                exampleTokens={currentComponent.data.example_tokens}
                exampleCi={currentComponent.data.example_ci}
                activatingTokens={inputTopRecall.map(({ token }) => token)}
            />
        {:else if currentComponent?.status === "error"}
            <StatusText>Error loading component data: {String(currentComponent.error)}</StatusText>
        {:else}
            <StatusText>Something went wrong loading component data.</StatusText>
        {/if}
    </div>
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

    .search-box {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .search-input {
        width: 70px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        appearance: textfield;
    }

    .search-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .search-input::-webkit-inner-spin-button,
    .search-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .search-input::placeholder {
        color: var(--text-muted);
    }

    .search-error {
        font-size: var(--text-xs);
        color: var(--semantic-error);
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
        padding: var(--space-4);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
    }

    .mean-ci {
        font-weight: 400;
        color: var(--text-muted);
        font-family: var(--font-mono);
        margin-left: var(--space-2);
    }

    .token-stats-row {
        display: flex;
        gap: var(--space-4);
    }

    .token-stats-row > :global(*) {
        flex: 1;
        min-width: 0;
    }

    .correlations-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .loading {
        padding: var(--space-4);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
