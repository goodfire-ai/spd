<script lang="ts">
    /**
     * Dataset attributions for a single component.
     *
     * Terminology:
     * - "Incoming" = sources that attribute TO this component (this component is the target)
     * - "Outgoing" = targets that this component attributes TO (this component is the source)
     */
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import type { DatasetAttributions } from "../../lib/useComponentData.svelte";
    import EdgeAttributionGrid from "./EdgeAttributionGrid.svelte";

    type Props = {
        attributions: DatasetAttributions;
        onComponentClick?: (componentKey: string) => void;
    };

    let { attributions, onComponentClick }: Props = $props();

    const PAGE_SIZE = 4;

    function handleClick(key: string) {
        if (onComponentClick) {
            onComponentClick(key);
        }
    }

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
</script>

<EdgeAttributionGrid
    title="Dataset Attributions"
    incomingLabel="Incoming"
    outgoingLabel="Outgoing"
    incomingPositive={positiveSources}
    incomingNegative={negativeSources}
    outgoingPositive={positiveTargets}
    outgoingNegative={negativeTargets}
    pageSize={PAGE_SIZE}
    onClick={handleClick}
/>
