<script lang="ts">
    import TokenHighlights from "./TokenHighlights.svelte";

    interface Props {
        // Columnar data
        exampleTokens: string[][]; // [n_examples, window_size]
        exampleCi: number[][]; // [n_examples, window_size]
        exampleActivePos: number[]; // [n_examples]
        // Unique activating tokens (from pr_tokens, already sorted by recall)
        activatingTokens: string[];
    }

    let { exampleTokens, exampleCi, exampleActivePos, activatingTokens }: Props = $props();

    let currentPage = $state(0);
    let pageSize = $state(20);
    let tokenFilter = $state<string | null>(null);

    // Number of examples (guard against null during transitions)
    let nExamples = $derived(exampleTokens?.length ?? 0);

    // Update currentPage when page input changes
    function handlePageInput(event: Event) {
        const { value } = event.target as HTMLInputElement;
        if (value === "") return;
        const valueNum = parseInt(value);
        if (!isNaN(valueNum) && valueNum >= 1 && valueNum <= totalPages) {
            currentPage = valueNum - 1;
        } else {
            alert("something went wrong");
            currentPage = 0;
        }
    }

    // Filter example indices by token
    let filteredIndices = $derived.by(() => {
        if (tokenFilter === null) {
            return Array.from({ length: nExamples }, (_, i) => i);
        }

        const indices: number[] = [];
        for (let i = 0; i < nExamples; i++) {
            const tokens = exampleTokens[i];
            const ci = exampleCi[i];
            for (let j = 0; j < tokens.length; j++) {
                if (tokens[j] === tokenFilter && ci[j] > 0) {
                    indices.push(i);
                    break;
                }
            }
        }
        return indices;
    });

    let paginatedIndices = $derived.by(() => {
        const start = currentPage * pageSize;
        const end = start + pageSize;
        return filteredIndices.slice(start, end);
    });

    let totalPages = $derived(Math.ceil(filteredIndices.length / pageSize));

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset to page 0 when data, page size, or filter changes
    $effect(() => {
        exampleTokens;
        pageSize;
        tokenFilter;
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
        <div class="filter-control">
            <label for="token-filter">Filter by token:</label>
            <select id="token-filter" bind:value={tokenFilter}>
                <option value="">All tokens</option>
                {#each activatingTokens as token (token)}
                    <option value={token}>{token}</option>
                {/each}
            </select>
        </div>
    </div>
    <div class="examples">
        {#each paginatedIndices as idx (idx)}
            <div class="example-item">
                <TokenHighlights
                    tokenStrings={exampleTokens[idx]}
                    tokenCi={exampleCi[idx]}
                    activePosition={exampleActivePos[idx]}
                />
            </div>
        {/each}
    </div>
</div>

<style>
    .container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .examples {
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        flex-wrap: wrap;
    }

    .filter-control,
    .page-size-control {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .filter-control label,
    .page-size-control label {
        font-size: 0.9rem;
        color: #495057;
        white-space: nowrap;
    }

    .filter-control select,
    .page-size-control select {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        font-size: 0.9rem;
        background: white;
        cursor: pointer;
        min-width: 120px;
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #f8f9fa;
    }

    .pagination button {
        padding: 0.25rem 0.75rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .pagination button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .pagination span {
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
        appearance: none;
        margin: 0;
    }

    .example-item {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.8;
        color: #333;
        padding: 0.5rem;
        overflow: visible;
    }
</style>
