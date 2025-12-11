<script lang="ts">
    import type { DatasetSearchResult } from "../lib/api";

    type Props = {
        results: DatasetSearchResult[];
        page: number;
        pageSize: number;
        totalPages: number;
        onPageChange: (page: number) => void;
        query: string;
    };

    let { results, page, pageSize, totalPages, onPageChange, query }: Props = $props();

    function highlightQuery(text: string, searchQuery: string): string {
        if (!searchQuery) return escapeHtml(text);
        // Escape regex special chars in the query
        const escapedQuery = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const regex = new RegExp(`(${escapedQuery})`, "gi");
        // Use placeholder markers, then escape HTML, then replace markers with mark tags
        const MARK_START = "\u0000MARK_START\u0000";
        const MARK_END = "\u0000MARK_END\u0000";
        const markedText = text.replace(regex, `${MARK_START}$1${MARK_END}`);
        return escapeHtml(markedText)
            .replace(new RegExp(MARK_START, "g"), "<mark>")
            .replace(new RegExp(MARK_END, "g"), "</mark>");
    }

    function escapeHtml(text: string): string {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function prevPage() {
        if (page > 1) onPageChange(page - 1);
    }

    function nextPage() {
        if (page < totalPages) onPageChange(page + 1);
    }
</script>

<div class="results-container">
    <div class="results-list">
        {#each results as result, idx (idx)}
            <div class="result-card">
                <div class="result-header">
                    <span class="result-index">#{(page - 1) * pageSize + idx + 1}</span>
                    <span class="occurrence-badge"
                        >{result.occurrence_count} occurrence{result.occurrence_count > 1 ? "s" : ""}</span
                    >
                    {#if result.topic}
                        <span class="tag topic">{result.topic}</span>
                    {/if}
                    {#if result.theme}
                        <span class="tag theme">{result.theme}</span>
                    {/if}
                </div>
                <!-- eslint-disable svelte/no-at-html-tags -- highlightQuery escapes HTML before inserting safe mark tags -->
                <div class="story-text">
                    {@html highlightQuery(result.story, query)}
                </div>
                <!-- eslint-enable svelte/no-at-html-tags -->
            </div>
        {/each}
    </div>

    {#if totalPages > 1}
        <div class="pagination">
            <button class="page-button" onclick={prevPage} disabled={page === 1}> Previous </button>
            <span class="page-info"> Page {page} of {totalPages} </span>
            <button class="page-button" onclick={nextPage} disabled={page === totalPages}> Next </button>
        </div>
    {/if}
</div>

<style>
    .results-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
        height: 100%;
    }

    .results-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        flex: 1;
        overflow-y: auto;
    }

    .result-card {
        padding: var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .result-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
    }

    .result-index {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        font-weight: 600;
    }

    .occurrence-badge {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
    }

    .tag {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
    }

    .tag.topic {
        background: var(--bg-inset);
        color: var(--text-secondary);
    }

    .tag.theme {
        background: var(--bg-inset);
        color: var(--text-secondary);
    }

    .story-text {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-primary);
        line-height: 1.6;
        white-space: pre-wrap;
    }

    .story-text :global(mark) {
        background: color-mix(in srgb, var(--accent-primary) 20%, transparent);
        color: var(--accent-primary);
        padding: 0 2px;
        font-weight: 600;
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
