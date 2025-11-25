<script lang="ts">
    import ActivationContextComponent from "./ActivationContext.svelte";

    interface Props {
        // Columnar data
        exampleTokens: string[][];      // [n_examples][window_size]
        exampleCi: number[][];          // [n_examples][window_size]
        exampleActivePos: number[];     // [n_examples]
        // Unique activating tokens (from pr_tokens, already sorted by recall)
        activatingTokens: string[];
    }

    let { exampleTokens, exampleCi, exampleActivePos, activatingTokens }: Props = $props();

    let currentPage = $state(0);
    let pageSize = $state(20);
    let tokenFilter = $state("");

    // Number of examples (guard against null during transitions)
    let nExamples = $derived(exampleTokens?.length ?? 0);

    // Update currentPage when page input changes
    function handlePageInput(event: Event) {
        const target = event.target as HTMLInputElement;
        const value = parseInt(target.value);
        if (!isNaN(value) && value >= 1 && value <= totalPages) {
            currentPage = value - 1;
        } else {
            alert("something went wrong");
            currentPage = 0;
        }
    }

    // Filter example indices by token
    let filteredIndices = $derived.by(() => {
        const startTime = performance.now();
        const indices: number[] = [];
        for (let i = 0; i < nExamples; i++) {
            if (!tokenFilter) {
                indices.push(i);
            } else {
                const tokens = exampleTokens[i];
                const ci = exampleCi[i];
                for (let j = 0; j < tokens.length; j++) {
                    if (tokens[j] === tokenFilter && ci[j] > 0) {
                        indices.push(i);
                        break;
                    }
                }
            }
        }
        console.log(`[timing] filteredIndices: ${(performance.now() - startTime).toFixed(2)}ms (${nExamples} examples)`);
        return indices;
    });

    let paginatedIndices = $derived.by(() => {
        const startTime = performance.now();
        const start = currentPage * pageSize;
        const end = start + pageSize;
        const result = filteredIndices.slice(start, end);
        console.log(`[timing] paginatedIndices: ${(performance.now() - startTime).toFixed(2)}ms (${result.length} items)`);
        return result;
    });

    // Debug: inspect reactive values
    $inspect("PagedTable", { nExamples, pageSize, currentPage, paginatedCount: paginatedIndices.length });

    // Measure render time using $effect.pre (before DOM) and $effect (after DOM)
    let renderStart = 0;
    let renderCount = 0;
    $effect.pre(() => {
        // Track dependencies - runs before DOM update
        exampleTokens;
        paginatedIndices;
        renderStart = performance.now();
        renderCount++;
        console.log(`[timing] PagedTable pre-render #${renderCount}`);
    });

    $effect(() => {
        // Runs after DOM update
        exampleTokens;
        paginatedIndices;
        if (renderStart > 0) {
            console.log(`[timing] PagedTable render #${renderCount}: ${(performance.now() - renderStart).toFixed(2)}ms (${paginatedIndices.length} items)`);
        }
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
        if (exampleTokens) currentPage = 0;
    });

    $effect(() => {
        if (pageSize) currentPage = 0;
    });

    $effect(() => {
        if (tokenFilter !== undefined) currentPage = 0;
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
            <ActivationContextComponent
                tokenStrings={exampleTokens[idx]}
                tokenCi={exampleCi[idx]}
                activePosition={exampleActivePos[idx]}
            />
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
</style>
