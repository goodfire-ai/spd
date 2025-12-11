<script lang="ts">
    import type { CorrelatedComponent } from "../../lib/localAttributionsTypes";
    import ComponentPillList from "../ui/ComponentPillList.svelte";

    type Props = {
        title: string;
        mathNotation?: string;
        items: CorrelatedComponent[];
        onComponentClick?: (componentKey: string) => void;
    };

    let { title, mathNotation, items, onComponentClick }: Props = $props();
</script>

{#if items.length > 0}
    <div class="correlation-column">
        <h5>
            {title}
            {#if mathNotation}
                <span class="math-notation">{mathNotation}</span>
            {/if}
        </h5>
        <ComponentPillList {items} {onComponentClick} />
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
</style>
