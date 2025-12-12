<script lang="ts">
    import * as api from "../lib/api";
    import DatasetSearchResults from "./DatasetSearchResults.svelte";

    let query = $state("");
    let split = $state<"train" | "test">("train");
    let loading = $state(false);
    let metadata = $state<api.DatasetSearchMetadata | null>(null);
    let currentPage = $state(1);
    let pageSize = $state(20);
    let currentPageResults = $state<api.DatasetSearchPage | null>(null);
    let error = $state<string | null>(null);

    async function performSearch() {
        if (!query.trim()) return;

        loading = true;
        metadata = null;
        currentPageResults = null;
        currentPage = 1;
        error = null;

        try {
            const result = await api.searchDataset(query.trim(), split);
            metadata = result;
            await loadPage(1);
        } finally {
            loading = false;
        }
    }

    async function loadPage(page: number) {
        currentPageResults = await api.getDatasetSearchPage(page, pageSize);
        currentPage = page;
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && !loading) {
            performSearch();
        }
    }
</script>

<div class="tab-content">
    <div class="config-box">
        <div class="config-header">
            <span class="config-title">Search SimpleStories Dataset</span>
            <button class="search-button" onclick={performSearch} disabled={loading || !query.trim()}>
                {loading ? "Searching..." : "Search"}
            </button>
        </div>
        <div class="search-form">
            <div class="form-row">
                <label for="query">Query:</label>
                <input
                    id="query"
                    type="text"
                    placeholder="e.g. 'dragon' or 'went to the'"
                    bind:value={query}
                    onkeydown={handleKeydown}
                    disabled={loading}
                />
            </div>
            <div class="form-row">
                <label for="split">Split:</label>
                <select id="split" bind:value={split} disabled={loading}>
                    <option value="train">Train</option>
                    <option value="test">Test</option>
                </select>
            </div>
        </div>
        {#if metadata}
            <div class="metadata">
                <span>Found {metadata.total_results} results in {metadata.search_time_seconds.toFixed(2)}s</span>
            </div>
        {/if}
        {#if error}
            <div class="error-banner">
                {error}
            </div>
        {/if}
    </div>

    <div class="results-box">
        {#if currentPageResults}
            <DatasetSearchResults
                results={currentPageResults.results}
                page={currentPage}
                {pageSize}
                totalPages={currentPageResults.total_pages}
                onPageChange={loadPage}
                {query}
            />
        {:else if loading}
            <div class="empty-state">Searching dataset...</div>
        {:else}
            <div class="empty-state">
                <p>No search performed yet</p>
                <p class="hint">Enter a query above to search the SimpleStories dataset</p>
            </div>
        {/if}
    </div>
</div>

<style>
    .tab-content {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        gap: var(--space-4);
        padding: var(--space-6);
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

    .search-form {
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

    .search-button {
        padding: var(--space-1) var(--space-3);
        border: none;
        background: var(--accent-primary);
        color: white;
        font-weight: 500;
        font-size: var(--text-sm);
    }

    .search-button:hover:not(:disabled) {
        background: var(--accent-primary-dim);
    }

    .search-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
    }

    .metadata {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .error-banner {
        background: var(--bg-surface);
        color: var(--status-negative);
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--status-negative);
        border-radius: var(--radius-md);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
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

    .empty-state .hint {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
</style>
