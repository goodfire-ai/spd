<script lang="ts">
    import type { ComponentCorrelations } from "../../lib/localAttributionsTypes";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import ComponentCorrelationPills from "../local-attr/ComponentCorrelationPills.svelte";

    type Props = {
        correlations: ComponentCorrelations;
        pageSize: number;
        onComponentClick?: (componentKey: string) => void;
    };

    let { correlations, pageSize, onComponentClick }: Props = $props();
</script>

<div class="correlations-grid">
    {#if displaySettings.isCorrelationStatVisible("pmi")}
        <ComponentCorrelationPills title="PMI" items={correlations.pmi} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                log(P(<span class="color-this">this</span>, <span class="color-that">that</span>) / P(<span
                    class="color-this">this</span
                >)P(<span class="color-that">that</span>))
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.isCorrelationStatVisible("bottom_pmi")}
        <ComponentCorrelationPills title="Bottom PMI" items={correlations.bottom_pmi} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                lowest PMI (anti-correlated)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.isCorrelationStatVisible("precision")}
        <ComponentCorrelationPills title="Predictors" items={correlations.precision} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                <span class="color-both">P</span>(<span class="color-this">this</span> |
                <span class="color-that">that</span>)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.isCorrelationStatVisible("recall")}
        <ComponentCorrelationPills title="Predictees" items={correlations.recall} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                <span class="color-both">P</span>(<span class="color-that">that</span> |
                <span class="color-this">this</span>)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.isCorrelationStatVisible("jaccard")}
        <ComponentCorrelationPills title="Jaccard" items={correlations.jaccard} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                <span class="color-both">this</span> ∩ <span class="color-both">that</span> / (<span class="color-this"
                    >this</span
                >
                ∪ <span class="color-that">that</span>)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
</div>

<style>
    .correlations-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: var(--space-4);
    }
</style>
