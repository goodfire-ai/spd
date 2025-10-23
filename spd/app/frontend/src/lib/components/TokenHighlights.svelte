<script lang="ts">
    export let tokenStrings: string[];
    export let tokenCiValues: number[];
    export let activePosition: number = -1;
    export let precision: number = 3;

    const getHighlightColor = (importance: number): string => {
        return `rgba(0, 200, 0, ${importance * 0.5})`;
    };

    function fmtTokenString(s: string): string {
        // handle wordpieces chunked tokenization
        if (s.startsWith("##")) {
            return s.slice(2);
        } else {
            return ` ${s}`;
        }
    }
</script>

<span class="token-highlights">
    {#each tokenStrings as tokenString, idx (idx)}
        {#if tokenCiValues[idx] > 0}
            <span
                class="token-highlight"
                class:active-token={idx === activePosition}
                style={`background-color:${getHighlightColor(tokenCiValues[idx])};`}
                data-ci={`CI: ${tokenCiValues[idx].toFixed(precision)}`}>{fmtTokenString(tokenString)}</span
            >
        {:else}
            {fmtTokenString(tokenString)}
        {/if}
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
