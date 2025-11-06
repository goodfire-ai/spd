<script lang="ts">
    import type { ActivationContext } from "../lib/api";
    import ActivationContextComponent from "./ActivationContext.svelte";

    interface Props {
        examples: ActivationContext[];
    }

    let { examples }: Props = $props();

    let currentPage = $state(0);
    let pageSize = $state(100);
    let tokenFilter = $state("");

    // Display page (1-indexed)
    let displayPage = $state(1);

    // Sync displayPage with currentPage
    $effect(() => {
        displayPage = currentPage + 1;
    });

    // Update currentPage when displayPage changes
    function handlePageInput(event: Event) {
        const target = event.target as HTMLInputElement;
        const value = parseInt(target.value);
        if (!isNaN(value) && value >= 1 && value <= totalPages) {
            currentPage = value - 1;
        } else {
            // Reset to current valid page if invalid
            displayPage = currentPage + 1;
        }
    }

    // Get unique tokens from all examples
    let allActivatingTokens = $derived.by(() => {
        const tokenSet = new Set<string>(
            examples.flatMap((example) =>
                example.token_strings.filter(
                    (_, idx) => example.token_ci_values[idx] > 0.01,
                ),
            ),
        );
        return Array.from(tokenSet).sort();
    });

    // Filter examples by token
    let filteredExamples = $derived.by(() => {
        if (!tokenFilter) return examples;
        return examples.filter((example) =>
            example.token_strings.some(
                (token, idx) =>
                    token === tokenFilter && example.token_ci_values[idx] > 0,
            ),
        );
    });

    let paginatedExamples = $derived.by(() => {
        const start = currentPage * pageSize;
        const end = start + pageSize;
        return filteredExamples.slice(start, end);
    });

    let totalPages = $derived(Math.ceil(filteredExamples.length / pageSize));

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset to page 0 when examples, page size, or filter changes
    $effect(() => {
        if (examples) currentPage = 0;
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
            <button onclick={previousPage} disabled={currentPage === 0}
                >&lt;</button
            >
            <input
                type="number"
                min="1"
                max={totalPages}
                value={displayPage}
                oninput={handlePageInput}
                class="page-input"
            />
            <span>of {totalPages}</span>
            <button onclick={nextPage} disabled={currentPage === totalPages - 1}
                >&gt;</button
            >
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
                {#each allActivatingTokens as token (token)}
                    <option value={token}>{token}</option>
                {/each}
            </select>
        </div>
    </div>
    <div class="examples">
        {#each paginatedExamples as example (example.__id)}
            <ActivationContextComponent {example} />
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
