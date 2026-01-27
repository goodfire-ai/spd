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
    import EdgeAttributionGrid from "./EdgeAttributionGrid.svelte";

    type Props = {
        attributions: DatasetAttributions;
        onComponentClick?: (componentKey: string) => void;
    };

    let { attributions, onComponentClick }: Props = $props();

    function handleClick(key: string) {
        if (onComponentClick) {
            onComponentClick(key);
        }
    }

    function toEdgeAttribution(
        entries: { component_key: string; value: number }[],
        maxAbsValue: number,
    ): EdgeAttribution[] {
        return entries.map((e) => ({
            key: e.component_key,
            value: e.value,
            normalizedMagnitude: Math.abs(e.value) / (maxAbsValue || 1),
        }));
    }

    const maxSourceVal = $derived(
        Math.max(attributions.positive_sources[0]?.value ?? 0, Math.abs(attributions.negative_sources[0]?.value ?? 0)),
    );
    const maxTargetVal = $derived(
        Math.max(attributions.positive_targets[0]?.value ?? 0, Math.abs(attributions.negative_targets[0]?.value ?? 0)),
    );

    const positiveSources = $derived(toEdgeAttribution(attributions.positive_sources, maxSourceVal));
    const negativeSources = $derived(toEdgeAttribution(attributions.negative_sources, maxSourceVal));
    const positiveTargets = $derived(toEdgeAttribution(attributions.positive_targets, maxTargetVal));
    const negativeTargets = $derived(toEdgeAttribution(attributions.negative_targets, maxTargetVal));
</script>

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
