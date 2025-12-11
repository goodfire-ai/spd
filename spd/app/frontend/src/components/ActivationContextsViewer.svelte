<script lang="ts">
    import type { ComponentDetail, HarvestMetadata } from "../lib/api";
    import * as api from "../lib/api";
    import { getComponentCorrelations, getComponentTokenStats } from "../lib/localAttributionsApi";
    import type { ComponentCorrelations, TokenStats } from "../lib/localAttributionsTypes";
    import { viewSettings } from "../lib/viewSettings.svelte";
    import ActivationContextsPagedTable from "./ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "./ComponentProbeInput.svelte";
    import ComponentCorrelationTable from "./local-attr/ComponentCorrelationTable.svelte";
    import SectionHeader from "./ui/SectionHeader.svelte";
    import StatusText from "./ui/StatusText.svelte";
    import TokenStatsSection from "./ui/TokenStatsSection.svelte";

    interface Props {
        harvestMetadata: HarvestMetadata;
    }

    let { harvestMetadata }: Props = $props();

    const N_TOKENS_TO_DISPLAY_INPUT = 80;
    const N_TOKENS_TO_DISPLAY_OUTPUT = 30;
    const N_CORRELATIONS_TO_DISPLAY = 20;

    let availableLayers = $derived(Object.keys(harvestMetadata.layers).sort());
    let currentPage = $state(0);
    let selectedLayer = $state<string>(Object.keys(harvestMetadata.layers)[0]);

    let componentCache = $state<Record<string, ComponentDetail>>({});
    let loadingComponent = $state(false);

    // Correlations state
    let correlations = $state<ComponentCorrelations | null>(null);
    let correlationsLoading = $state(false);

    // Token stats state (from batch job)
    let tokenStats = $state<TokenStats | null>(null);
    let tokenStatsLoading = $state(false);

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
        getComponentCorrelations(layer, cIdx, N_CORRELATIONS_TO_DISPLAY)
            .then((data) => {
                correlations = data;
            })
            .finally(() => {
                correlationsLoading = false;
            });
    });

    // Fetch token stats when component changes (from batch job)
    $effect(() => {
        const layer = selectedLayer;
        const cIdx = currentMetadata?.subcomponent_idx;
        if (cIdx === undefined) return;

        tokenStats = null;
        tokenStatsLoading = true;
        getComponentTokenStats(layer, cIdx, Math.max(N_TOKENS_TO_DISPLAY_INPUT, N_TOKENS_TO_DISPLAY_OUTPUT))
            .then((data) => {
                tokenStats = data;
            })
            .finally(() => {
                tokenStatsLoading = false;
            });
    });

    // === Input token stats (what tokens activate this component) ===
    let inputTopRecall = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.input.top_recall.map(([token, value]) => ({ token, value }));
    });

    let inputTopPmi = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.input.top_pmi.map(([token, value]) => ({ token, value }));
    });

    // === Output token stats (what tokens this component predicts) ===
    let outputTopPmi = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.output.top_pmi.map(([token, value]) => ({ token, value }));
    });

    let outputBottomPmi = $derived.by(() => {
        if (!tokenStats) return [];
        return tokenStats.output.bottom_pmi.map(([token, value]) => ({ token, value }));
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
    </div>

    {#if loadingComponent}
        <div class="loading">Loading component data...</div>
    {:else if currentComponent && currentMetadata}
        <div class="component-section">
            <SectionHeader title="Subcomponent {currentMetadata.subcomponent_idx}" level="h4">
                <span class="mean-ci">Mean CI: {formatMeanCi(currentMetadata.mean_ci)}</span>
            </SectionHeader>

            <div class="token-stats-row">
                <TokenStatsSection
                    sectionTitle="Input Tokens"
                    sectionSubtitle="(what activates this component)"
                    loading={tokenStatsLoading}
                    lists={[
                        {
                            title: "Top PMI",
                            mathNotation: "log(P(firing, token) / P(firing)P(token))",
                            items: inputTopPmi.slice(0, N_TOKENS_TO_DISPLAY_INPUT),
                        },
                    ]}
                />

                <TokenStatsSection
                    sectionTitle="Output Tokens"
                    sectionSubtitle="(what this component predicts)"
                    loading={tokenStatsLoading}
                    lists={[
                        {
                            title: "Top PMI",
                            mathNotation: "positive association with predictions",
                            items: outputTopPmi.slice(0, N_TOKENS_TO_DISPLAY_OUTPUT),
                        },
                        {
                            title: "Bottom PMI",
                            mathNotation: "negative association with predictions",
                            items: outputBottomPmi.slice(0, N_TOKENS_TO_DISPLAY_OUTPUT),
                        },
                    ]}
                />
            </div>

            <ComponentProbeInput layer={selectedLayer} componentIdx={currentMetadata.subcomponent_idx} />

            <!-- Component correlations -->
            <div class="correlations-section">
                <SectionHeader title="Correlated Components" />
                {#if correlations}
                    <div class="correlations-grid">
                        {#if viewSettings.isCorrelationStatVisible("pmi")}
                            <ComponentCorrelationTable
                                title="PMI"
                                mathNotation="log(P(both) / P(A)P(B))"
                                items={correlations.pmi}
                                maxItems={N_CORRELATIONS_TO_DISPLAY}
                            />
                        {/if}
                        {#if viewSettings.isCorrelationStatVisible("precision")}
                            <ComponentCorrelationTable
                                title="Precision"
                                mathNotation="P(that | this)"
                                items={correlations.precision}
                                maxItems={N_CORRELATIONS_TO_DISPLAY}
                            />
                        {/if}
                        {#if viewSettings.isCorrelationStatVisible("recall")}
                            <ComponentCorrelationTable
                                title="Recall"
                                mathNotation="P(this | that)"
                                items={correlations.recall}
                                maxItems={N_CORRELATIONS_TO_DISPLAY}
                            />
                        {/if}
                        {#if viewSettings.isCorrelationStatVisible("f1")}
                            <ComponentCorrelationTable
                                title="F1"
                                items={correlations.f1}
                                maxItems={N_CORRELATIONS_TO_DISPLAY}
                            />
                        {/if}
                        {#if viewSettings.isCorrelationStatVisible("jaccard")}
                            <ComponentCorrelationTable
                                title="Jaccard"
                                items={correlations.jaccard}
                                maxItems={N_CORRELATIONS_TO_DISPLAY}
                            />
                        {/if}
                    </div>
                {:else if correlationsLoading}
                    <StatusText>Loading...</StatusText>
                {:else if correlationJobStatus === null}
                    <StatusText>No correlations data. Use "Harvest" in the top bar to compute.</StatusText>
                {:else if correlationJobStatus.status === "pending"}
                    <StatusText variant="pending">Job {correlationJobStatus.job_id} pending...</StatusText>
                {:else if correlationJobStatus.status === "running"}
                    <StatusText variant="running">Job {correlationJobStatus.job_id} running...</StatusText>
                {:else if correlationJobStatus.status === "failed"}
                    <StatusText variant="failed">Correlation job failed. Check top bar to retry.</StatusText>
                {:else}
                    <StatusText>No correlations for this component.</StatusText>
                {/if}
            </div>

            <ActivationContextsPagedTable
                exampleTokens={currentComponent.example_tokens}
                exampleCi={currentComponent.example_ci}
                exampleActivePos={currentComponent.example_active_pos}
                activatingTokens={inputTopRecall.map(({ token }) => token)}
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
