<script lang="ts">
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import EdgeAttributionList from "./EdgeAttributionList.svelte";
    import SectionHeader from "./SectionHeader.svelte";

    type Props = {
        title: string;
        incomingLabel: string;
        outgoingLabel: string;
        incomingPositive: EdgeAttribution[];
        incomingNegative: EdgeAttribution[];
        outgoingPositive: EdgeAttribution[];
        outgoingNegative: EdgeAttribution[];
        pageSize: number;
        onClick: (key: string) => void;
    };

    let {
        title,
        incomingLabel,
        outgoingLabel,
        incomingPositive,
        incomingNegative,
        outgoingPositive,
        outgoingNegative,
        pageSize,
        onClick,
    }: Props = $props();

    const hasAnyIncoming = $derived(incomingPositive.length > 0 || incomingNegative.length > 0);
    const hasAnyOutgoing = $derived(outgoingPositive.length > 0 || outgoingNegative.length > 0);
</script>

{#if hasAnyIncoming}
    <div class="edge-list-group">
        <SectionHeader title="{title} – {incomingLabel}" />
        <div class="pos-neg-row">
            {#if incomingPositive.length > 0}
                <div class="edge-list">
                    <EdgeAttributionList
                        items={incomingPositive}
                        {pageSize}
                        {onClick}
                        direction="positive"
                        title="Positive"
                    />
                </div>
            {/if}
            {#if incomingNegative.length > 0}
                <div class="edge-list">
                    <EdgeAttributionList
                        items={incomingNegative}
                        {pageSize}
                        {onClick}
                        direction="negative"
                        title="Negative"
                    />
                </div>
            {/if}
        </div>
    </div>
{/if}

{#if hasAnyOutgoing}
    <div class="edge-list-group">
        <SectionHeader title="{title} – {outgoingLabel}" />
        <div class="pos-neg-row">
            {#if outgoingPositive.length > 0}
                <div class="edge-list">
                    <EdgeAttributionList
                        items={outgoingPositive}
                        {pageSize}
                        {onClick}
                        direction="positive"
                        title="Positive"
                    />
                </div>
            {/if}
            {#if outgoingNegative.length > 0}
                <div class="edge-list">
                    <EdgeAttributionList
                        items={outgoingNegative}
                        {pageSize}
                        {onClick}
                        direction="negative"
                        title="Negative"
                    />
                </div>
            {/if}
        </div>
    </div>
{/if}

<style>
    .edge-list-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .pos-neg-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-3);
    }

    .edge-list {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }
</style>
