<script lang="ts">
    import { getContext } from "svelte";
    import * as api from "../lib/api";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import DatasetSearchResults from "./DatasetSearchResults.svelte";
    import TokenizedSampleCard from "./TokenizedSampleCard.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type InnerTab = "search" | "random";
    let activeInnerTab = $state<InnerTab>("search");

    // Search state
    let searchQuery = $state("");
    let searchSplit = $state<"train" | "test">("train");
    let searchLoading = $state(false);
    let searchMetadata = $state<api.DatasetSearchMetadata | null>(null);
    let searchPage = $state(1);
    let searchPageSize = $state(20);
    let searchResults = $state<api.DatasetSearchPage | null>(null);

    // Random samples state (text-only, no run required)
    let randomSplit = $state<"train" | "test">("train");
    let randomSeed = $state(42);
    let randomLoading = $state(false);
    let randomData = $state<api.RandomSamplesResult | null>(null);
    let randomPage = $state(1);
    let randomPageSize = $state(20);

    // Random samples with CE loss state (requires loaded run)
    let randomWithLossData = $state<api.RandomSamplesWithLossResult | null>(null);
    let randomWithLossLoading = $state(false);
    let randomWithLossPage = $state(1);
    let randomWithLossPageSize = $state(5);

    // Check if run is loaded
    let isRunLoaded = $derived(runState.run.status === "loaded");

    async function performSearch() {
        if (!searchQuery.trim()) return;

        searchLoading = true;
        searchMetadata = null;
        searchResults = null;
        searchPage = 1;

        try {
            const result = await api.searchDataset(searchQuery.trim(), searchSplit);
            searchMetadata = result;
            await loadSearchPage(1);
        } finally {
            searchLoading = false;
        }
    }

    async function loadSearchPage(page: number) {
        searchResults = await api.getDatasetResults(page, searchPageSize);
        searchPage = page;
    }

    function handleSearchKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && !searchLoading) {
            performSearch();
        }
    }

    async function loadRandomSamples() {
        if (isRunLoaded) {
            // Use new endpoint with CE loss
            randomWithLossLoading = true;
            randomWithLossData = null;
            randomWithLossPage = 1;
            try {
                randomWithLossData = await api.getRandomSamplesWithLoss(20, randomSeed, randomSplit);
            } finally {
                randomWithLossLoading = false;
            }
        } else {
            // Use old endpoint (text only)
            randomLoading = true;
            randomData = null;
            randomPage = 1;
            try {
                randomData = await api.getRandomSamples(100, randomSeed, randomSplit);
            } finally {
                randomLoading = false;
            }
        }
    }

    function shuffleRandom() {
        randomSeed = Math.floor(Math.random() * 10000);
        loadRandomSamples();
    }

    // Pagination for random samples without CE loss (client-side)
    let randomPageResults = $derived.by(() => {
        if (!randomData) return null;
        const start = (randomPage - 1) * randomPageSize;
        const end = start + randomPageSize;
        return {
            results: randomData.results.slice(start, end),
            page: randomPage,
            page_size: randomPageSize,
            total_results: randomData.results.length,
            total_pages: Math.ceil(randomData.results.length / randomPageSize),
        };
    });

    // Pagination for random samples with CE loss (client-side)
    let randomWithLossPageResults = $derived.by(() => {
        if (!randomWithLossData) return null;
        const start = (randomWithLossPage - 1) * randomWithLossPageSize;
        const end = start + randomWithLossPageSize;
        return {
            results: randomWithLossData.results.slice(start, end),
            page: randomWithLossPage,
            total_results: randomWithLossData.results.length,
            total_pages: Math.ceil(randomWithLossData.results.length / randomWithLossPageSize),
        };
    });

    function handleRandomPageChange(page: number) {
        randomPage = page;
    }

    function handleRandomWithLossPageChange(page: number) {
        randomWithLossPage = page;
    }

    // Derived loading state for random samples
    let anyRandomLoading = $derived(randomLoading || randomWithLossLoading);
</script>

<div class="explorer-container">
    <div class="inner-tabs">
        <button
            type="button"
            class="inner-tab"
            class:active={activeInnerTab === "search"}
            onclick={() => (activeInnerTab = "search")}
        >
            Search
        </button>
        <button
            type="button"
            class="inner-tab"
            class:active={activeInnerTab === "random"}
            onclick={() => (activeInnerTab = "random")}
        >
            Random Samples
        </button>
    </div>

    <div class="tab-panels">
        <!-- Search Panel -->
        <div class="tab-panel" class:hidden={activeInnerTab !== "search"}>
            <div class="config-box">
                <div class="config-header">
                    <span class="config-title">Search SimpleStories Dataset</span>
                    <button
                        class="action-button"
                        onclick={performSearch}
                        disabled={searchLoading || !searchQuery.trim()}
                    >
                        {searchLoading ? "Searching..." : "Search"}
                    </button>
                </div>
                <div class="form-grid">
                    <div class="form-row">
                        <label for="search-query">Query:</label>
                        <input
                            id="search-query"
                            type="text"
                            placeholder="e.g. 'dragon' or 'went to the'"
                            bind:value={searchQuery}
                            onkeydown={handleSearchKeydown}
                            disabled={searchLoading}
                        />
                    </div>
                    <div class="form-row">
                        <label for="search-split">Split:</label>
                        <select id="search-split" bind:value={searchSplit} disabled={searchLoading}>
                            <option value="train">Train</option>
                            <option value="test">Test</option>
                        </select>
                    </div>
                </div>
                {#if searchMetadata}
                    <div class="metadata">
                        Found {searchMetadata.total_results} results in {searchMetadata.search_time_seconds.toFixed(2)}s
                    </div>
                {/if}
            </div>

            <div class="results-box">
                {#if searchResults}
                    <DatasetSearchResults
                        results={searchResults.results}
                        page={searchPage}
                        pageSize={searchPageSize}
                        totalPages={searchResults.total_pages}
                        onPageChange={loadSearchPage}
                        query={searchQuery}
                    />
                {:else if searchLoading}
                    <div class="empty-state">Searching dataset...</div>
                {:else}
                    <div class="empty-state">
                        <p>Enter a query above to search the SimpleStories dataset</p>
                    </div>
                {/if}
            </div>
        </div>

        <!-- Random Samples Panel -->
        <div class="tab-panel" class:hidden={activeInnerTab !== "random"}>
            <div class="config-box">
                <div class="config-header">
                    <span class="config-title">Random Dataset Samples</span>
                    <div class="action-buttons">
                        <button class="action-button" onclick={loadRandomSamples} disabled={anyRandomLoading}>
                            {anyRandomLoading ? "Loading..." : "Load Samples"}
                        </button>
                        <button class="action-button secondary" onclick={shuffleRandom} disabled={anyRandomLoading}>
                            Shuffle
                        </button>
                    </div>
                </div>
                <div class="form-grid">
                    <div class="form-row">
                        <label for="random-split">Split:</label>
                        <select id="random-split" bind:value={randomSplit} disabled={anyRandomLoading}>
                            <option value="train">Train</option>
                            <option value="test">Test</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="random-seed">Seed:</label>
                        <input
                            id="random-seed"
                            type="number"
                            bind:value={randomSeed}
                            disabled={anyRandomLoading}
                            min="0"
                        />
                    </div>
                </div>
                {#if isRunLoaded && randomWithLossData}
                    <div class="metadata">
                        Showing {randomWithLossData.results.length} random samples with CE loss (seed: {randomWithLossData.seed})
                    </div>
                {:else if randomData}
                    <div class="metadata">
                        Showing {randomData.results.length} random samples (seed: {randomData.seed})
                    </div>
                {/if}
                {#if !isRunLoaded}
                    <div class="metadata muted">Load a run to see per-token cross-entropy loss</div>
                {/if}
            </div>

            <div class="results-box">
                {#if isRunLoaded && randomWithLossPageResults}
                    <!-- Tokenized samples with next-token probability -->
                    <div class="tokenized-results">
                        <div class="prob-legend">
                            <span class="legend-label">P(next token):</span>
                            <span class="legend-low">Low</span>
                            <span class="legend-arrow">→</span>
                            <span class="legend-high">High</span>
                        </div>
                        <div class="tokenized-list">
                            {#each randomWithLossPageResults.results as sample, idx (idx)}
                                <TokenizedSampleCard
                                    {sample}
                                    index={(randomWithLossPage - 1) * randomWithLossPageSize + idx}
                                />
                            {/each}
                        </div>
                        {#if randomWithLossPageResults.total_pages > 1}
                            <div class="pagination">
                                <button
                                    class="page-button"
                                    onclick={() => handleRandomWithLossPageChange(randomWithLossPage - 1)}
                                    disabled={randomWithLossPage === 1}
                                >
                                    Previous
                                </button>
                                <span class="page-info">
                                    Page {randomWithLossPage} of {randomWithLossPageResults.total_pages}
                                </span>
                                <button
                                    class="page-button"
                                    onclick={() => handleRandomWithLossPageChange(randomWithLossPage + 1)}
                                    disabled={randomWithLossPage === randomWithLossPageResults.total_pages}
                                >
                                    Next
                                </button>
                            </div>
                        {/if}
                    </div>
                {:else if !isRunLoaded && randomPageResults}
                    <!-- Text-only samples (no run loaded) -->
                    <DatasetSearchResults
                        results={randomPageResults.results}
                        page={randomPage}
                        pageSize={randomPageSize}
                        totalPages={randomPageResults.total_pages}
                        onPageChange={handleRandomPageChange}
                        query=""
                    />
                {:else if anyRandomLoading}
                    <div class="empty-state">Loading random samples...</div>
                {:else}
                    <div class="empty-state">
                        <p>Click "Load Samples" to fetch random stories</p>
                    </div>
                {/if}
            </div>
        </div>
    </div>
</div>

<style>
    .explorer-container {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        padding: var(--space-6);
        gap: var(--space-4);
    }

    .inner-tabs {
        display: flex;
        gap: var(--space-1);
        border-bottom: 1px solid var(--border-default);
        padding-bottom: var(--space-2);
    }

    .inner-tab {
        padding: var(--space-2) var(--space-4);
        background: none;
        border: none;
        border-bottom: 2px solid transparent;
        font: inherit;
        font-weight: 500;
        font-size: var(--text-sm);
        color: var(--text-muted);
        cursor: pointer;
        transition:
            color var(--transition-normal),
            border-color var(--transition-normal);
    }

    .inner-tab:hover {
        color: var(--text-primary);
    }

    .inner-tab.active {
        color: var(--text-primary);
        border-bottom-color: var(--accent-primary);
    }

    .tab-panels {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-panel {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
    }

    .tab-panel.hidden {
        display: none;
    }

    .config-box {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
    }

    .config-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .config-title {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .action-buttons {
        display: flex;
        gap: var(--space-2);
    }

    .action-button {
        padding: var(--space-1) var(--space-3);
        border: none;
        background: var(--accent-primary);
        color: white;
        font-weight: 500;
        font-size: var(--text-sm);
        cursor: pointer;
    }

    .action-button:hover:not(:disabled) {
        background: var(--accent-primary-dim);
    }

    .action-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .action-button.secondary {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }

    .action-button.secondary:hover:not(:disabled) {
        background: var(--bg-surface);
        color: var(--text-primary);
    }

    .form-grid {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .form-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .form-row label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
        min-width: 60px;
    }

    .form-row input,
    .form-row select {
        flex: 1;
        max-width: 400px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
    }

    .form-row input:focus,
    .form-row select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .metadata {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .results-box {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
        overflow-y: auto;
    }

    .empty-state {
        display: flex;
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        text-align: center;
        font-family: var(--font-sans);
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
    }

    .metadata.muted {
        color: var(--text-muted);
        font-style: italic;
    }

    .tokenized-results {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        height: 100%;
    }

    .prob-legend {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .legend-label {
        color: var(--text-secondary);
        font-weight: 600;
    }

    .legend-low {
        color: var(--text-muted);
    }

    .legend-arrow {
        color: var(--text-muted);
    }

    .legend-high {
        color: rgb(22, 163, 74);
    }

    .tokenized-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        flex: 1;
        overflow-y: auto;
    }

    .pagination {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--space-3);
        padding: var(--space-3) 0;
        border-top: 1px solid var(--border-default);
    }

    .page-button {
        padding: var(--space-1) var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-weight: 500;
        font-size: var(--text-sm);
    }

    .page-button:hover:not(:disabled) {
        background: var(--accent-primary);
        color: white;
        border-color: var(--accent-primary);
    }

    .page-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .page-info {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }
</style>
