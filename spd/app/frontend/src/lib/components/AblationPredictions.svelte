<script lang="ts">
    import TokenPredictions from "./TokenPredictions.svelte";
    import AblationHeatmap from "./AblationHeatmap.svelte";
    import type { OutputTokenLogit, ComponentMask, MaskOverride, AblationStats } from "$lib/api";

    export let tokenLogits: OutputTokenLogit[][];
    export let promptTokens: string[];
    export let appliedMask: ComponentMask;
    export let maskOverride: MaskOverride | undefined = undefined;
    export let ablationStats: AblationStats;
</script>

<div class="ablation-output-section">
    <TokenPredictions {tokenLogits} {promptTokens} containerClass="" {appliedMask} />

    <AblationHeatmap {ablationStats} {promptTokens} />

    <div class="ablation-summary">
        <h3>Applied ablations:</h3>
        <div class="applied-ablations">
            {#if maskOverride}
                <div class="mask-override-info">
                    <strong>Mask Override Applied:</strong>
                    <span class="mask-description"
                        >{maskOverride.description || "Unnamed mask"}</span
                    >
                    <span class="mask-details">
                        (Layer: {maskOverride.layer}, L0: {maskOverride.combined_mask.l0})
                    </span>
                </div>
            {:else}
                {#each Object.entries(appliedMask) as [layerName, tokenArrays]}
                    {#each tokenArrays as disabledComponents, tokenIdx}
                        {#if disabledComponents.length > 0}
                            <div class="applied-ablation-item">
                                <strong>{promptTokens[tokenIdx]}</strong>
                                in
                                <em>{layerName}</em>: disabled components {disabledComponents.join(
                                    ", "
                                )}
                            </div>
                        {/if}
                    {/each}
                {/each}
            {/if}
        </div>
    </div>
</div>

<style>
    .ablation-output-section {
        margin-top: 1rem;
        padding: 0.75rem;
        border: 2px solid #ff6b35;
        border-radius: 8px;
        background-color: #fff8f5;
    }

    .ablation-summary {
        background: white;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #ddd;
        margin-top: 0.5rem;
    }

    .ablation-summary h3 {
        margin: 0 0 0.25rem 0;
        color: #333;
    }

    .applied-ablations {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .applied-ablation-item {
        padding: 0.25rem;
        background-color: #f8f9fa;
        border-radius: 4px;
        border-left: 3px solid #ff6b35;
        font-size: 0.85rem;
        color: #555;
    }

    .mask-override-info {
        padding: 0.5rem;
        background: #e3f2fd;
        border-left: 3px solid #2196f3;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    .mask-description {
        font-weight: 600;
        color: #1976d2;
    }

    .mask-details {
        color: #666;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
</style>
