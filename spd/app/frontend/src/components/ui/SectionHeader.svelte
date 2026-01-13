<script lang="ts">
    import type { Snippet } from "svelte";

    type Props = {
        title: string;
        subtitle?: string;
        mathNotation?: string;
        /** Use h4 (larger) or h5 (smaller) */
        level?: "h4" | "h5";
        children?: Snippet;
    };

    let { title, subtitle, mathNotation, level = "h5", children }: Props = $props();
</script>

{#if level === "h4"}
    <h4>
        {title}
        {#if subtitle}
            <span class="subtitle">{subtitle}</span>
        {/if}
        {#if mathNotation}
            <span class="math-notation">{mathNotation}</span>
        {/if}
        {#if children}
            {@render children()}
        {/if}
    </h4>
{:else}
    <h5>
        {title}
        {#if subtitle}
            <span class="subtitle">{subtitle}</span>
        {/if}
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

    .subtitle {
        font-weight: 400;
        color: var(--text-muted);
        font-size: var(--text-xs);
        margin-left: var(--space-1);
    }

    .math-notation {
        font-family: var(--font-mono);
        font-style: normal;
        font-size: var(--text-xs);
        font-weight: 400;
        opacity: 0.7;
    }
</style>
