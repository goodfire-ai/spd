<script lang="ts">
    import { displaySettings, EXAMPLE_COLOR_MODE_LABELS, type ExampleColorMode } from "../lib/displaySettings.svelte";
    import TokenHighlights from "./TokenHighlights.svelte";

    interface Props {
        // Columnar data
        exampleTokens: string[][]; // [n_examples, window_size]
        exampleCi: number[][]; // [n_examples, window_size]
        exampleComponentActs: number[][]; // [n_examples, window_size]
        // Unique activating tokens (from pr_tokens, already sorted by recall)
        activatingTokens: string[];
        // Global max for normalization
        maxAbsComponentAct: number;
    }

    let { exampleTokens, exampleCi, exampleComponentActs, activatingTokens, maxAbsComponentAct }: Props = $props();

    let currentPage = $state(0);
    let pageSize = $state(20);
    let tokenFilter = $state<string | null>(null);

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
        exampleTokens; // eslint-disable-line @typescript-eslint/no-unused-expressions
        pageSize; // eslint-disable-line @typescript-eslint/no-unused-expressions
        tokenFilter; // eslint-disable-line @typescript-eslint/no-unused-expressions
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
            <label for="token-filter">Filter by includes token:</label>
            <select id="token-filter" bind:value={tokenFilter}>
                <option value="">All tokens</option>
                {#each activatingTokens as token (token)}
                    <option value={token}>{token}</option>
                {/each}
            </select>
        </div>
        <div class="color-mode-control">
            <label for="color-mode-select">Color by:</label>
            <select
                id="color-mode-select"
                value={displaySettings.exampleColorMode}
                onchange={(e) => (displaySettings.exampleColorMode = e.currentTarget.value as ExampleColorMode)}
            >
                {#each Object.entries(EXAMPLE_COLOR_MODE_LABELS) as [mode, label] (mode)}
                    <option value={mode}>{label}</option>
                {/each}
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
                        colorMode={displaySettings.exampleColorMode}
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

    .filter-control,
    .page-size-control,
    .color-mode-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .filter-control label,
    .page-size-control label,
    .color-mode-control label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        white-space: nowrap;
        font-weight: 500;
    }

    .filter-control select,
    .page-size-control select,
    .color-mode-control select {
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

    .filter-control select:focus,
    .page-size-control select:focus,
    .color-mode-control select:focus {
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
