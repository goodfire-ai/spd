<script lang="ts">
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import type { DatasetAttributions } from "../../lib/useComponentData.svelte";
    import EdgeAttributionList from "./EdgeAttributionList.svelte";
    import SectionHeader from "./SectionHeader.svelte";

    type Props = {
        attributions: DatasetAttributions;
        onComponentClick?: (componentKey: string) => void;
    };

    let { attributions, onComponentClick }: Props = $props();

    const PAGE_SIZE = 10;

    function handleClick(key: string) {
        if (onComponentClick) {
            onComponentClick(key);
        }
    }

    // Prepare display data with normalized magnitudes
    function toEdgeAttribution(
        entries: { componentKey: string; value: number }[],
        maxAbsValue: number,
    ): EdgeAttribution[] {
        return entries.map((e) => ({
            key: e.componentKey,
            value: e.value,
            normalizedMagnitude: Math.abs(e.value) / (maxAbsValue || 1),
        }));
    }

    // Max abs values for scaling - computed from all sources/targets
    const maxSourceVal = $derived(
        Math.max(attributions.positiveSources[0]?.value ?? 0, Math.abs(attributions.negativeSources[0]?.value ?? 0)),
    );
    const maxTargetVal = $derived(
        Math.max(attributions.positiveTargets[0]?.value ?? 0, Math.abs(attributions.negativeTargets[0]?.value ?? 0)),
    );

    const positiveSources = $derived(toEdgeAttribution(attributions.positiveSources, maxSourceVal));
    const negativeSources = $derived(toEdgeAttribution(attributions.negativeSources, maxSourceVal));
    const positiveTargets = $derived(toEdgeAttribution(attributions.positiveTargets, maxTargetVal));
    const negativeTargets = $derived(toEdgeAttribution(attributions.negativeTargets, maxTargetVal));

    const hasAnySources = $derived(positiveSources.length > 0 || negativeSources.length > 0);
    const hasAnyTargets = $derived(positiveTargets.length > 0 || negativeTargets.length > 0);
    const hasAnyData = $derived(hasAnySources || hasAnyTargets);
</script>

{#if hasAnyData}
    <div class="dataset-attributions-section">
        <SectionHeader title="Dataset Attributions" />
        <div class="edge-lists-grid">
            {#if hasAnySources}
                <div class="edge-list-group">
                    <h5>Incoming (sources)</h5>
                    {#if positiveSources.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Positive</span>
                            <EdgeAttributionList
                                items={positiveSources}
                                pageSize={PAGE_SIZE}
                                onClick={handleClick}
                                direction="positive"
                            />
                        </div>
                    {/if}
                    {#if negativeSources.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Negative</span>
                            <EdgeAttributionList
                                items={negativeSources}
                                pageSize={PAGE_SIZE}
                                onClick={handleClick}
                                direction="negative"
                            />
                        </div>
                    {/if}
                </div>
            {/if}

            {#if hasAnyTargets}
                <div class="edge-list-group">
                    <h5>Outgoing (targets)</h5>
                    {#if positiveTargets.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Positive</span>
                            <EdgeAttributionList
                                items={positiveTargets}
                                pageSize={PAGE_SIZE}
                                onClick={handleClick}
                                direction="positive"
                            />
                        </div>
                    {/if}
                    {#if negativeTargets.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Negative</span>
                            <EdgeAttributionList
                                items={negativeTargets}
                                pageSize={PAGE_SIZE}
                                onClick={handleClick}
                                direction="negative"
                            />
                        </div>
                    {/if}
                </div>
            {/if}
        </div>
    </div>
{/if}

<style>
    .dataset-attributions-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .edge-lists-grid {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-4);
    }

    .edge-list-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .edge-list-group h5 {
        margin: 0;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .edge-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .edge-list-title {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-style: italic;
    }
</style>
