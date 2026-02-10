<script lang="ts">
    import { SvelteSet } from "svelte/reactivity";
    import type { TokenizedSearchResult } from "../lib/api/dataset";
    import { getNextTokenProbBgColor } from "../lib/colors";

    interface Props {
        result: TokenizedSearchResult;
        index: number;
        query: string;
    }

    let { result, index, query }: Props = $props();

    // Shift by 1: position i shows probability that token i-1 predicted token i
    function getProbAtPosition(i: number): number | null {
        if (i === 0) return null;
        return result.next_token_probs[i - 1];
    }

    function formatProb(prob: number | null): string {
        if (prob === null) return "";
        return `P(token): ${(prob * 100).toFixed(1)}%`;
    }

    // Check if a token is part of a query match by building the string and checking positions
    const matchPositions = $derived.by(() => {
        if (!query) return new SvelteSet<number>();

        const positions = new SvelteSet<number>();
        const lowerQuery = query.toLowerCase();

        // Rebuild the text and track character positions for each token
        let text = "";
        const tokenStarts: number[] = [];
        const tokenEnds: number[] = [];

        for (const tok of result.tokens) {
            tokenStarts.push(text.length);
            text += tok;
            tokenEnds.push(text.length);
        }

        const lowerText = text.toLowerCase();

        // Find all occurrences of query in text
        let searchStart = 0;
        while (true) {
            const matchStart = lowerText.indexOf(lowerQuery, searchStart);
            if (matchStart === -1) break;
            const matchEnd = matchStart + lowerQuery.length;

            // Mark all tokens that overlap with this match
            for (let i = 0; i < result.tokens.length; i++) {
                if (tokenStarts[i] < matchEnd && tokenEnds[i] > matchStart) {
                    positions.add(i);
                }
            }

            searchStart = matchStart + 1;
        }

        return positions;
    });
</script>

<div class="result-card">
    <div class="result-header">
        <span class="result-index">#{index + 1}</span>
        {#if result.occurrence_count > 0}
            <span class="occurrence-badge"
                >{result.occurrence_count} occurrence{result.occurrence_count !== 1 ? "s" : ""}</span
            >
        {/if}
        {#each Object.entries(result.metadata) as [metaKey, metaVal] (metaKey)}
            <span class="tag">{metaVal}</span>
        {/each}
    </div>
    <div class="tokens-container">
        <span class="prob-tokens"
            >{#each result.tokens as tok, i (i)}<span
                    class="prob-token"
                    class:match={matchPositions.has(i)}
                    style="background-color: {getNextTokenProbBgColor(getProbAtPosition(i))}"
                    data-tooltip={formatProb(getProbAtPosition(i))}>{tok}</span
                >{/each}</span
        >
    </div>
</div>

<style>
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
        background: var(--bg-inset);
        color: var(--text-secondary);
    }

    .tokens-container {
        font-size: var(--text-sm);
        line-height: 1.6;
    }

    .prob-tokens {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 1px;
        font-family: var(--font-mono);
    }

    .prob-token {
        padding: 1px 2px;
        border: 1px solid transparent;
        position: relative;
        white-space: pre;
    }

    .prob-token.match {
        border-color: var(--accent-primary);
        border-radius: 2px;
    }

    .prob-token::after {
        content: attr(data-tooltip);
        position: absolute;
        top: calc(100% + 4px);
        left: 0;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 1000;
    }

    .prob-token:hover::after {
        opacity: 1;
    }
</style>
