<script lang="ts">
    import { getNextTokenProbBgColor } from "../lib/colors";

    interface Props {
        tokens: string[];
        nextTokenProbs: (number | null)[];
    }

    let { tokens, nextTokenProbs }: Props = $props();

    // Shift by 1: position i shows probability that token i-1 predicted token i
    function getProbAtPosition(i: number): number | null {
        if (i === 0) return null;
        return nextTokenProbs[i - 1];
    }

    function formatProb(prob: number | null): string {
        if (prob === null) return "";
        return `P(token): ${(prob * 100).toFixed(1)}%`;
    }
</script>

<span class="prob-tokens"
    >{#each tokens as tok, i (i)}<span
            class="prob-token"
            style="background-color: {getNextTokenProbBgColor(getProbAtPosition(i))}"
            data-tooltip={formatProb(getProbAtPosition(i))}>{tok}</span
        >{/each}</span
>

<style>
    .prob-tokens {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 1px;
        font-family: var(--font-mono);
    }

    .prob-token {
        padding: 1px 2px;
        border-right: 1px solid var(--border-subtle);
        position: relative;
        white-space: pre;
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
