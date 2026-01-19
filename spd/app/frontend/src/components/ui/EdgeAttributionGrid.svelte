<script lang="ts">
    import type { EdgeAttribution, OutputProbEntry } from "../../lib/promptAttributionsTypes";
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
        // Optional: only needed for prompt-level attributions with wte/output pseudo-layers
        tokens?: string[];
        outputProbs?: Record<string, OutputProbEntry>;
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
        tokens,
        outputProbs,
    }: Props = $props();

    const hasAnyIncoming = $derived(incomingPositive.length > 0 || incomingNegative.length > 0);
    const hasAnyOutgoing = $derived(outgoingPositive.length > 0 || outgoingNegative.length > 0);
    const hasAnyData = $derived(hasAnyIncoming || hasAnyOutgoing);
</script>

{#if hasAnyData}
    <div class="attribution-grid-section">
        <SectionHeader {title} />
        <div class="edge-lists-grid">
            {#if hasAnyIncoming}
                <div class="edge-list-group">
                    <h5>{incomingLabel}</h5>
                    {#if incomingPositive.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Positive</span>
                            <EdgeAttributionList
                                items={incomingPositive}
                                {pageSize}
                                {onClick}
                                direction="positive"
                                {tokens}
                                {outputProbs}
                            />
                        </div>
                    {/if}
                    {#if incomingNegative.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Negative</span>
                            <EdgeAttributionList
                                items={incomingNegative}
                                {pageSize}
                                {onClick}
                                direction="negative"
                                {tokens}
                                {outputProbs}
                            />
                        </div>
                    {/if}
                </div>
            {/if}

            {#if hasAnyOutgoing}
                <div class="edge-list-group">
                    <h5>{outgoingLabel}</h5>
                    {#if outgoingPositive.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Positive</span>
                            <EdgeAttributionList
                                items={outgoingPositive}
                                {pageSize}
                                {onClick}
                                direction="positive"
                                {tokens}
                                {outputProbs}
                            />
                        </div>
                    {/if}
                    {#if outgoingNegative.length > 0}
                        <div class="edge-list">
                            <span class="edge-list-title">Negative</span>
                            <EdgeAttributionList
                                items={outgoingNegative}
                                {pageSize}
                                {onClick}
                                direction="negative"
                                {tokens}
                                {outputProbs}
                            />
                        </div>
                    {/if}
                </div>
            {/if}
        </div>
    </div>
{/if}

<style>
    .attribution-grid-section {
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
