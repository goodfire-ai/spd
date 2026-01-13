<script lang="ts">
    import type { GlobalAttributions } from "../../lib/api/correlations";
    import GlobalAttributionsList from "./GlobalAttributionsList.svelte";
    import SectionHeader from "./SectionHeader.svelte";

    type Props = {
        data: GlobalAttributions;
        pageSize?: number;
        onComponentClick?: (componentKey: string) => void;
    };

    let { data, pageSize = 10, onComponentClick }: Props = $props();

    const hasSources = $derived(
        data.top_positive_sources.length > 0 || data.top_negative_sources.length > 0,
    );
    const hasTargets = $derived(
        data.top_positive_targets.length > 0 || data.top_negative_targets.length > 0,
    );
</script>

<div class="global-attributions">
    {#if hasSources}
        <div class="direction-group">
            <SectionHeader title="Top Sources" subtitle="Components that attribute TO this component" />
            <div class="attribution-columns">
                {#if data.top_positive_sources.length > 0}
                    <div class="column">
                        <h6 class="column-header positive">Positive</h6>
                        <GlobalAttributionsList
                            items={data.top_positive_sources}
                            {pageSize}
                            signColor="positive"
                            {onComponentClick}
                        />
                    </div>
                {/if}
                {#if data.top_negative_sources.length > 0}
                    <div class="column">
                        <h6 class="column-header negative">Negative</h6>
                        <GlobalAttributionsList
                            items={data.top_negative_sources}
                            {pageSize}
                            signColor="negative"
                            {onComponentClick}
                        />
                    </div>
                {/if}
            </div>
        </div>
    {/if}

    {#if hasTargets}
        <div class="direction-group">
            <SectionHeader title="Top Targets" subtitle="Components this component attributes TO" />
            <div class="attribution-columns">
                {#if data.top_positive_targets.length > 0}
                    <div class="column">
                        <h6 class="column-header positive">Positive</h6>
                        <GlobalAttributionsList
                            items={data.top_positive_targets}
                            {pageSize}
                            signColor="positive"
                            {onComponentClick}
                        />
                    </div>
                {/if}
                {#if data.top_negative_targets.length > 0}
                    <div class="column">
                        <h6 class="column-header negative">Negative</h6>
                        <GlobalAttributionsList
                            items={data.top_negative_targets}
                            {pageSize}
                            signColor="negative"
                            {onComponentClick}
                        />
                    </div>
                {/if}
            </div>
        </div>
    {/if}

    {#if !hasSources && !hasTargets}
        <p class="no-data">No global attribution data available for this component.</p>
    {/if}
</div>

<style>
    .global-attributions {
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
    }

    .direction-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .attribution-columns {
        display: flex;
        gap: var(--space-4);
    }

    .column {
        flex: 1;
        min-width: 0;
    }

    .column-header {
        margin: 0 0 var(--space-1) 0;
        font-size: var(--text-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .column-header.positive {
        color: rgb(22, 163, 74);
    }

    .column-header.negative {
        color: rgb(220, 38, 38);
    }

    .no-data {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-style: italic;
        margin: 0;
    }
</style>
