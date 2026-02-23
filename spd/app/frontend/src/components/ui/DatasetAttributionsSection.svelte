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
    import type { AttrMetric } from "../../lib/api/datasetAttributions";
    import EdgeAttributionGrid from "./EdgeAttributionGrid.svelte";

    const METRIC_LABELS: Record<AttrMetric, string> = {
        attr: "Signed",
        attr_abs: "Abs Target",
        mean_squared_attr: "RMS",
    };

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

    const active = $derived(attributions[selectedMetric]);

    function toEdgeAttribution(
        entries: { component_key: string; value: number; token_str: string | null }[],
        maxAbsValue: number,
    ): EdgeAttribution[] {
        return entries.map((e) => ({
            key: e.component_key,
            value: e.value,
            normalizedMagnitude: Math.abs(e.value) / (maxAbsValue || 1),
            tokenStr: e.token_str,
        }));
    }

    const maxSourceVal = $derived(
        Math.max(
            active.positive_sources[0]?.value ?? 0,
            Math.abs(active.negative_sources[0]?.value ?? 0),
        ),
    );
    const maxTargetVal = $derived(
        Math.max(
            active.positive_targets[0]?.value ?? 0,
            Math.abs(active.negative_targets[0]?.value ?? 0),
        ),
    );

    const hasSigned = $derived(selectedMetric === "attr");

    const positiveSources = $derived(toEdgeAttribution(active.positive_sources, maxSourceVal));
    const negativeSources = $derived(hasSigned ? toEdgeAttribution(active.negative_sources, maxSourceVal) : []);
    const positiveTargets = $derived(toEdgeAttribution(active.positive_targets, maxTargetVal));
    const negativeTargets = $derived(hasSigned ? toEdgeAttribution(active.negative_targets, maxTargetVal) : []);
</script>

<div class="section">
    <div class="metric-selector">
        {#each Object.entries(METRIC_LABELS) as [metric, label] (metric)}
            <label class="radio-item">
                <input
                    type="radio"
                    name="dataset-attr-metric"
                    checked={selectedMetric === metric}
                    onchange={() => (selectedMetric = metric as AttrMetric)}
                />
                <span class="stat-label">{label}</span>
            </label>
        {/each}
    </div>

    <EdgeAttributionGrid
        title="Dataset Attributions"
        incomingLabel="Incoming"
        outgoingLabel="Outgoing"
        incomingPositive={positiveSources}
        incomingNegative={negativeSources}
        outgoingPositive={positiveTargets}
        outgoingNegative={negativeTargets}
        pageSize={COMPONENT_CARD_CONSTANTS.DATASET_ATTRIBUTIONS_PAGE_SIZE}
        onClick={handleClick}
    />
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
