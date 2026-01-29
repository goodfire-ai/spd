<script lang="ts">
    import * as api from "../lib/api";
    import type { Loadable } from "../lib";
    import DatasetSearchResults from "./DatasetSearchResults.svelte";
    import TokenizedSampleCard from "./TokenizedSampleCard.svelte";

    type InnerTab = "search" | "random";
    let activeInnerTab = $state<InnerTab>("search");

    // Search state
    let searchQuery = $state("");
    let searchSplit = $state<"train" | "test">("train");
    let searchMetadata = $state<api.DatasetSearchMetadata | null>(null);
    let searchPage = $state(1);
    let searchPageSize = $state(10);
    let searchResults = $state<Loadable<api.TokenizedSearchPage>>({ status: "uninitialized" });

    // Random samples state
    let randomSplit = $state<"train" | "test">("train");
    let randomSeed = $state(42);
    let randomPage = $state(1);
    let randomPageSize = $state(5);
    let randomSamples = $state<Loadable<api.RandomSamplesWithLossResult>>({ status: "uninitialized" });

    async function performSearch() {
        if (!searchQuery.trim()) return;

        searchResults = { status: "loading" };
        searchMetadata = null;
        searchPage = 1;

        try {
            const result = await api.searchDataset(searchQuery.trim(), searchSplit);
            searchMetadata = result;
            await loadSearchPage(1);
        } catch (e) {
            searchResults = { status: "error", error: e };
        }
    }

    async function loadSearchPage(page: number) {
        searchResults = { status: "loading" };
        try {
            const data = await api.getTokenizedResults(page, searchPageSize);
            searchResults = { status: "loaded", data };
            searchPage = page;
        } catch (e) {
            searchResults = { status: "error", error: e };
        }
    }

    function handleSearchKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && searchResults.status !== "loading") {
            performSearch();
        }
    }

    async function loadRandomSamples() {
        randomSamples = { status: "loading" };
        randomPage = 1;
        try {
            const data = await api.getRandomSamplesWithLoss(20, randomSeed, randomSplit);
            randomSamples = { status: "loaded", data };
        } catch (e) {
            randomSamples = { status: "error", error: e };
        }
    }

    function shuffleRandom() {
        randomSeed = Math.floor(Math.random() * 10000);
        loadRandomSamples();
    }

    // Pagination for random samples (client-side)
    let randomPageResults = $derived.by(() => {
        if (randomSamples.status !== "loaded") return null;
        const start = (randomPage - 1) * randomPageSize;
        const end = start + randomPageSize;
        return {
            results: randomSamples.data.results.slice(start, end),
            page: randomPage,
            total_results: randomSamples.data.results.length,
            total_pages: Math.ceil(randomSamples.data.results.length / randomPageSize),
        };
    });

    function handleRandomPageChange(page: number) {
        randomPage = page;
    }
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
                        disabled={searchResults.status === "loading" || !searchQuery.trim()}
                    >
                        {searchResults.status === "loading" ? "Searching..." : "Search"}
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
                            disabled={searchResults.status === "loading"}
                        />
                    </div>
                    <div class="form-row">
                        <label for="search-split">Split:</label>
                        <select id="search-split" bind:value={searchSplit} disabled={searchResults.status === "loading"}>
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
                {#if searchResults.status === "loaded"}
                    <DatasetSearchResults
                        results={searchResults.data.results}
                        page={searchPage}
                        pageSize={searchPageSize}
                        totalPages={searchResults.data.total_pages}
                        onPageChange={loadSearchPage}
                        query={searchQuery}
                    />
                {:else if searchResults.status === "loading"}
                    <div class="empty-state">Searching dataset...</div>
                {:else if searchResults.status === "error"}
                    <div class="empty-state error">Error: {searchResults.error}</div>
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
                        <button class="action-button" onclick={loadRandomSamples} disabled={randomSamples.status === "loading"}>
                            {randomSamples.status === "loading" ? "Loading..." : "Load Samples"}
                        </button>
                        <button class="action-button secondary" onclick={shuffleRandom} disabled={randomSamples.status === "loading"}>
                            Shuffle
                        </button>
                    </div>
                </div>
                <div class="form-grid">
                    <div class="form-row">
                        <label for="random-split">Split:</label>
                        <select id="random-split" bind:value={randomSplit} disabled={randomSamples.status === "loading"}>
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
                            disabled={randomSamples.status === "loading"}
                            min="0"
                        />
                    </div>
                </div>
                {#if randomSamples.status === "loaded"}
                    <div class="metadata">
                        Showing {randomSamples.data.results.length} random samples (seed: {randomSamples.data.seed})
                    </div>
                {/if}
            </div>

            <div class="results-box">
                {#if randomPageResults}
                    <!-- Tokenized samples with next-token probability -->
                    <div class="tokenized-results">
                        <div class="prob-legend">
                            <span class="legend-label">P(token):</span>
                            <span class="legend-low">Low</span>
                            <span class="legend-arrow">→</span>
                            <span class="legend-high">High</span>
                        </div>
                        <div class="tokenized-list">
                            {#each randomPageResults.results as sample, idx (idx)}
                                <TokenizedSampleCard
                                    {sample}
                                    index={(randomPage - 1) * randomPageSize + idx}
                                />
                            {/each}
                        </div>
                        {#if randomPageResults.total_pages > 1}
                            <div class="pagination">
                                <button
                                    class="page-button"
                                    onclick={() => handleRandomPageChange(randomPage - 1)}
                                    disabled={randomPage === 1}
                                >
                                    Previous
                                </button>
                                <span class="page-info">
                                    Page {randomPage} of {randomPageResults.total_pages}
                                </span>
                                <button
                                    class="page-button"
                                    onclick={() => handleRandomPageChange(randomPage + 1)}
                                    disabled={randomPage === randomPageResults.total_pages}
                                >
                                    Next
                                </button>
                            </div>
                        {/if}
                    </div>
                {:else if randomSamples.status === "loading"}
                    <div class="empty-state">Loading random samples...</div>
                {:else if randomSamples.status === "error"}
                    <div class="empty-state error">Error: {randomSamples.error}</div>
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

    .empty-state.error {
        color: var(--status-negative);
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
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
