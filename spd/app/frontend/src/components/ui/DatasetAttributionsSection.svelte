<script lang="ts">
    /**
     * Dataset attributions for a single component.
     *
     * Terminology:
     * - "Incoming" = sources that attribute TO this component (this component is the target)
     * - "Outgoing" = targets that this component attributes TO (this component is the source)
     */
    import { COMPONENT_CARD_CONSTANTS } from "../../lib/componentCardConstants";
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import type { DatasetAttributions } from "../../lib/useComponentData.svelte";
    import type { AttrMetric, DatasetAttributionEntry } from "../../lib/api/datasetAttributions";
    import EdgeAttributionGrid from "./EdgeAttributionGrid.svelte";

    type Props = {
        attributions: DatasetAttributions;
        onComponentClick?: (componentKey: string) => void;
    };

    let { attributions, onComponentClick }: Props = $props();
    let selectedMetric = $state<AttrMetric>("attr");

    function handleClick(key: string) {
        if (onComponentClick) {
            onComponentClick(key);
        }
    }

    function toEdgeAttribution(entries: DatasetAttributionEntry[], maxAbsValue: number): EdgeAttribution[] {
        return entries.map((e) => ({
            key: e.component_key,
            value: e.value,
            normalizedMagnitude: Math.abs(e.value) / (maxAbsValue || 1),
            tokenStr: e.token_str,
        }));
    }

    function maxAbs(...vals: number[]): number {
        return Math.max(...vals.map(Math.abs));
    }

    // attr: signed
    const attrMaxSource = $derived(
        maxAbs(attributions.attr.positive_sources[0]?.value ?? 0, attributions.attr.negative_sources[0]?.value ?? 0),
    );
    const attrMaxTarget = $derived(
        maxAbs(attributions.attr.positive_targets[0]?.value ?? 0, attributions.attr.negative_targets[0]?.value ?? 0),
    );

    // attr_abs: signed
    const absMaxSource = $derived(
        maxAbs(
            attributions.attr_abs.positive_sources[0]?.value ?? 0,
            attributions.attr_abs.negative_sources[0]?.value ?? 0,
        ),
    );
    const absMaxTarget = $derived(
        maxAbs(
            attributions.attr_abs.positive_targets[0]?.value ?? 0,
            attributions.attr_abs.negative_targets[0]?.value ?? 0,
        ),
    );
</script>

<div class="section">
    <div class="metric-selector">
        <label class="radio-item">
            <input
                type="radio"
                name="dataset-attr-metric"
                checked={selectedMetric === "attr"}
                onchange={() => (selectedMetric = "attr")}
            />
            <span class="stat-label">Signed</span>
        </label>
        <label class="radio-item">
            <input
                type="radio"
                name="dataset-attr-metric"
                checked={selectedMetric === "attr_abs"}
                onchange={() => (selectedMetric = "attr_abs")}
            />
            <span class="stat-label">Abs Target</span>
        </label>
    </div>

    {#if selectedMetric === "attr"}
        <EdgeAttributionGrid
            title="Dataset Attributions"
            incomingLabel="Incoming"
            outgoingLabel="Outgoing"
            incomingPositive={toEdgeAttribution(attributions.attr.positive_sources, attrMaxSource)}
            incomingNegative={toEdgeAttribution(attributions.attr.negative_sources, attrMaxSource)}
            outgoingPositive={toEdgeAttribution(attributions.attr.positive_targets, attrMaxTarget)}
            outgoingNegative={toEdgeAttribution(attributions.attr.negative_targets, attrMaxTarget)}
            pageSize={COMPONENT_CARD_CONSTANTS.DATASET_ATTRIBUTIONS_PAGE_SIZE}
            onClick={handleClick}
        />
    {:else if selectedMetric === "attr_abs"}
        <EdgeAttributionGrid
            title="Dataset Attributions"
            incomingLabel="Incoming"
            outgoingLabel="Outgoing"
            incomingPositive={toEdgeAttribution(attributions.attr_abs.positive_sources, absMaxSource)}
            incomingNegative={toEdgeAttribution(attributions.attr_abs.negative_sources, absMaxSource)}
            outgoingPositive={toEdgeAttribution(attributions.attr_abs.positive_targets, absMaxTarget)}
            outgoingNegative={toEdgeAttribution(attributions.attr_abs.negative_targets, absMaxTarget)}
            pageSize={COMPONENT_CARD_CONSTANTS.DATASET_ATTRIBUTIONS_PAGE_SIZE}
            onClick={handleClick}
        />
    {/if}
</div>

<style>
    .section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .metric-selector {
        display: flex;
        gap: var(--space-3);
        font-size: var(--text-sm);
    }

    .radio-item {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        cursor: pointer;
        padding: var(--space-1);
        border-radius: var(--radius-sm);
    }

    .radio-item:hover {
        background: var(--bg-inset);
    }

    .radio-item input {
        margin: 0;
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .stat-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }
</style>
