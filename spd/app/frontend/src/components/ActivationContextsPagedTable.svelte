<script lang="ts">
    import { displaySettings } from "../lib/displaySettings.svelte";
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

    let examplesEl = $state<HTMLDivElement | undefined>(undefined);
    let currentPage = $state(0);
    let pageSize = $state(10);

    let nExamples = $derived(exampleTokens.length);

    function argmax(arr: number[]): number {
        let maxIdx = 0;
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    let firingPositions = $derived(exampleCi.map(argmax));

    // Minimum container width (in ch) so that per-row flex centering works without clipping.
    // Each row needs: 2 * max(leftWidth, rightWidth) + centerWidth.
    // Each token adds ~0.3ch overhead for border + margin beyond its character width.
    const TOKEN_OVERHEAD_CH = 0.3;

    let minWidthCh = $derived.by(() => {
        if (!displaySettings.centerOnPeak) return 0;
        let max = 0;
        for (let i = 0; i < exampleTokens.length; i++) {
            const fp = firingPositions[i];
            const tokens = exampleTokens[i];

            let leftWidth = 0;
            for (let j = 0; j < fp; j++) leftWidth += tokens[j].length + TOKEN_OVERHEAD_CH;

            let rightWidth = 0;
            for (let j = fp + 1; j < tokens.length; j++) rightWidth += tokens[j].length + TOKEN_OVERHEAD_CH;

            const centerWidth = tokens[fp].length + TOKEN_OVERHEAD_CH;
            const required = 2 * Math.max(leftWidth, rightWidth) + centerWidth;
            if (required > max) max = required;
        }
        return Math.ceil(max + 1);
    });

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

    function centerScroll() {
        if (!examplesEl) return;
        examplesEl.scrollLeft = (examplesEl.scrollWidth - examplesEl.clientWidth) / 2;
    }

    $effect(() => {
        if (!displaySettings.centerOnPeak) return;
        paginatedIndices; // eslint-disable-line @typescript-eslint/no-unused-expressions
        requestAnimationFrame(centerScroll);
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
        <label class="center-toggle">
            <input type="checkbox" bind:checked={displaySettings.centerOnPeak} />
            Center on peak
        </label>
    </div>
    <div class="examples" bind:this={examplesEl}>
        {#if displaySettings.centerOnPeak}
            <div class="examples-inner" style="min-width: {minWidthCh}ch">
                {#each paginatedIndices as idx (idx)}
                    {@const fp = firingPositions[idx]}
                    <div class="example-row">
                        <div class="left-tokens">
                            <TokenHighlights
                                tokenStrings={exampleTokens[idx].slice(0, fp)}
                                tokenCi={exampleCi[idx].slice(0, fp)}
                                tokenComponentActs={exampleComponentActs[idx].slice(0, fp)}
                                {maxAbsComponentAct}
                            />
                        </div>
                        <div class="center-token">
                            <TokenHighlights
                                tokenStrings={[exampleTokens[idx][fp]]}
                                tokenCi={[exampleCi[idx][fp]]}
                                tokenComponentActs={[exampleComponentActs[idx][fp]]}
                                {maxAbsComponentAct}
                            />
                        </div>
                        <div class="right-tokens">
                            <TokenHighlights
                                tokenStrings={exampleTokens[idx].slice(fp + 1)}
                                tokenCi={exampleCi[idx].slice(fp + 1)}
                                tokenComponentActs={exampleComponentActs[idx].slice(fp + 1)}
                                {maxAbsComponentAct}
                            />
                        </div>
                    </div>
                {/each}
            </div>
        {:else}
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
        {/if}
    </div>
</div>

<style>
    .container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .examples {
        padding: var(--space-2);
        overflow-x: auto;
        overflow-y: clip;
    }

    .examples-inner {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        min-width: 100%;
    }

    .example-row {
        display: flex;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }

    .example-item {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }

    .left-tokens {
        flex: 1 1 0;
        min-width: 0;
        text-align: right;
    }

    .center-token {
        flex: 0 0 auto;
    }

    .right-tokens {
        flex: 1 1 0;
        min-width: 0;
        text-align: left;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-2);
        border-bottom: 1px solid var(--border-default);
        flex-wrap: wrap;
    }

    .center-toggle {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
        cursor: pointer;
        margin-left: auto;
    }

    .center-toggle input {
        cursor: pointer;
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
</style>
