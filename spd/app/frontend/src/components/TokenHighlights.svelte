<script lang="ts">
    import { getTokenHighlightBg } from "../lib/colors";

    interface Props {
        tokenStrings: string[];
        tokenCi: number[]; // CI values (0-1 floats)
        activePosition: number;
    }

    let { tokenStrings, tokenCi, activePosition }: Props = $props();
</script>

<span class="token-highlights"
    >{#each tokenStrings as tok, i (i)}<span
            class="token-highlight"
            class:active-token={i === activePosition}
            style="background-color:{getTokenHighlightBg(tokenCi[i])}"
            data-ci="CI: {tokenCi[i].toFixed(3)}">{tok}</span
        >{/each}</span
>

<style>
    .token-highlights {
        display: inline;
        white-space: pre-wrap;
        font-family: var(--font-mono);
    }

    .token-highlight {
        display: inline;
        padding: 1px 0;
        margin-right: 1px;
        border-right: 1px solid var(--border-subtle);
        position: relative;
        white-space: pre;
    }

    .token-highlight::after {
        content: attr(data-ci);
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

    .token-highlight:hover::after {
        opacity: 1;
    }

    .token-highlight.active-token {
        outline: 1px solid var(--accent-primary);
        outline-offset: 1px;
    }
</style>
