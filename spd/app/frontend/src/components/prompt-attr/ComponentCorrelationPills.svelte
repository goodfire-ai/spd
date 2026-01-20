<script lang="ts">
    import type { Snippet } from "svelte";
    import type { CorrelatedComponent } from "../../lib/promptAttributionsTypes";
    import { colors } from "../../lib/colors";
    import CorrelatedComponentsList from "../ui/CorrelatedComponentsList.svelte";

    type Props = {
        title: string;
        mathNotation?: Snippet;
        items: CorrelatedComponent[];
        pageSize: number;
        onComponentClick?: (componentKey: string) => void;
    };

    let { title, mathNotation, items, pageSize, onComponentClick }: Props = $props();
</script>

{#if items.length > 0}
    <div
        class="correlation-column"
        style="--color-this: {colors.setOverlap.self}; --color-that: {colors.setOverlap.other}; --color-both: {colors
            .setOverlap.both};"
    >
        <h5>
            {title}
            {#if mathNotation}
                <span class="math-notation">{@render mathNotation()}</span>
            {/if}
        </h5>
        <CorrelatedComponentsList {items} {onComponentClick} {pageSize} />
    </div>
{/if}

<style>
    .correlation-column {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .correlation-column h5 {
        margin: 0;
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .correlation-column h5 .math-notation {
        font-weight: 400;
        font-style: italic;
        color: var(--text-muted);
        margin-left: var(--space-1);
    }

    /* Colors for math notation - match SetOverlapVis */
    .math-notation :global(.color-this) {
        color: var(--color-this);
    }

    .math-notation :global(.color-that) {
        color: var(--color-that);
    }

    .math-notation :global(.color-both) {
        color: var(--color-both);
    }
</style>
