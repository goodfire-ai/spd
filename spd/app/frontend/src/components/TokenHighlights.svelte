<script lang="ts">
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
            style="background-color:rgba(0,200,0,{tokenCi[i] * 0.5})"
            data-ci="CI: {tokenCi[i].toFixed(3)}">{tok}</span
        >{/each}</span
>

<style>
    .token-highlights {
        display: inline;
        white-space: pre-wrap;
    }

    .token-highlight {
        display: inline;
        padding: 2px 0px;
        border-radius: 2px;
        position: relative;
        white-space: pre;
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
