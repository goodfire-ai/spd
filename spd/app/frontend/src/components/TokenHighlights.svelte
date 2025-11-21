<script lang="ts">
    interface Props {
        tokenStrings: string[];
        tokenCiValues: number[];
        activePosition?: number;
        precision?: number;
    }

    let { tokenStrings, tokenCiValues, activePosition = -1, precision = 3 }: Props = $props();

    const getHighlightColor = (importance: number): string => {
        return `rgba(0, 200, 0, ${importance * 0.5})`;
    };
</script>

<span class="token-highlights">
    {#each tokenStrings as tokenString, idx (idx)}
        <span
            class="token-highlight"
            class:active-token={idx === activePosition}
            style={`background-color:${getHighlightColor(tokenCiValues[idx])};`}
            data-ci={`CI: ${tokenCiValues[idx].toFixed(precision)}`}>{tokenString}</span
        >
    {/each}
</span>

<style>
    .token-highlights {
        display: inline;
        white-space: pre-wrap;
    }

    .token-highlight {
        display: inline;
        padding: 2px 4px;
        border-radius: 3px;
        position: relative;
    }

    .token-highlight::after {
        content: attr(data-ci);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0s;
        margin-bottom: 4px;
        z-index: 1000;
    }

    .token-highlight:hover::after {
        opacity: 1;
    }

    .token-highlight.active-token {
        border: 2px solid rgba(255, 100, 0, 0.6);
    }
</style>
