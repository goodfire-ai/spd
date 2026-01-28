<script lang="ts">
    import TokenHighlights from "./TokenHighlights.svelte";

    interface Props {
        // Columnar data
        exampleTokens: string[][]; // [n_examples, window_size]
        exampleCi: number[][]; // [n_examples, window_size]
        exampleComponentActs: number[][]; // [n_examples, window_size]
        // Global max for normalization
        maxAbsComponentAct: number;
    }

    let { exampleTokens, exampleCi, exampleComponentActs, maxAbsComponentAct }: Props = $props();

    let currentPage = $state(0);
    let pageSize = $state(10);

    let nExamples = $derived(exampleTokens.length);

    // Update currentPage when page input changes
    function handlePageInput(event: Event) {
        const { value } = event.target as HTMLInputElement;
        if (value === "") return;
        const valueNum = parseInt(value);
        if (!isNaN(valueNum) && valueNum >= 1 && valueNum <= totalPages) {
            currentPage = valueNum - 1;
        } else {
            throw new Error(`Invalid page number: ${value} (must be 1-${totalPages})`);
        }
    }

    let allIndices = $derived(Array.from({ length: nExamples }, (_, i) => i));

    let paginatedIndices = $derived.by(() => {
        const start = currentPage * pageSize;
        const end = start + pageSize;
        return allIndices.slice(start, end);
    });

    let totalPages = $derived(Math.ceil(allIndices.length / pageSize));

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset to page 0 when data or page size changes
    $effect(() => {
        exampleTokens; // eslint-disable-line @typescript-eslint/no-unused-expressions
        pageSize; // eslint-disable-line @typescript-eslint/no-unused-expressions
        currentPage = 0;
    });
</script>

<div class="container">
    <div class="controls">
        <div class="pagination">
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
        <div class="page-size-control">
            <label for="page-size">Per page:</label>
            <select id="page-size" bind:value={pageSize}>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
            </select>
        </div>
    </div>
    <div class="examples">
        <div class="examples-inner">
            {#each paginatedIndices as idx (idx)}
                <div class="example-item">
                    <TokenHighlights
                        tokenStrings={exampleTokens[idx]}
                        tokenCi={exampleCi[idx]}
                        tokenComponentActs={exampleComponentActs[idx]}
                        {maxAbsComponentAct}
                    />
                </div>
            {/each}
        </div>
    </div>
</div>

<style>
    .container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .examples {
        padding: var(--space-2);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        overflow-x: auto;
        overflow-y: clip;
    }

    .examples-inner {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        width: max-content;
        min-width: 100%;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        flex-wrap: wrap;
    }

    .page-size-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .page-size-control label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        white-space: nowrap;
        font-weight: 500;
    }

    .page-size-control select {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 100px;
    }

    .page-size-control select:focus {
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
        background: var(--bg-inset);
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
        border-color: var(--accent-primary-dim);
    }

    .page-input::-webkit-inner-spin-button,
    .page-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .example-item {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }
</style>
