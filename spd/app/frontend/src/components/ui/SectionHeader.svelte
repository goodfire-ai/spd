<script lang="ts">
    import type { Snippet } from "svelte";

    type Props = {
        title: string;
        mathNotation?: string;
        /** Use h4 (larger) or h5 (smaller) */
        level?: "h4" | "h5";
        compact?: boolean;
        children?: Snippet;
    };

    let { title, mathNotation, level = "h5", compact = false, children }: Props = $props();
</script>

{#if level === "h4"}
    <h4 class:compact>
        {title}
        {#if mathNotation}
            <span class="math-notation">{mathNotation}</span>
        {/if}
        {#if children}
            {@render children()}
        {/if}
    </h4>
{:else}
    <h5 class:compact>
        {title}
        {#if mathNotation}
            <span class="math-notation">{mathNotation}</span>
        {/if}
        {#if children}
            {@render children()}
        {/if}
    </h5>
{/if}

<style>
    h4,
    h5 {
        margin: 0;
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    h4 {
        font-size: var(--text-sm);
    }

    h5 {
        font-size: var(--text-sm);
    }

    h4.compact {
        font-size: var(--text-xs);
    }

    h5.compact {
        font-size: var(--text-xs);
    }

    .math-notation {
        font-family: var(--font-mono);
        font-style: normal;
        font-size: var(--text-xs);
        font-weight: 400;
        opacity: 0.7;
    }
</style>
