<script lang="ts">
    import type { OutputTokenLogit, ComponentMask } from "$lib/api";
    import { isScrolling } from "$lib/stores/componentState";

    export let tokenLogits: OutputTokenLogit[][];
    export let promptTokens: string[];
    export let containerClass: string = "";
    export let appliedMask: ComponentMask | null = null;

    function hasAblation(tokenIdx: number): boolean {
        if (!appliedMask) return false;
        return Object.values(appliedMask).some(
            tokenMasks => tokenMasks[tokenIdx] && tokenMasks[tokenIdx].length > 0
        );
    }

    function getProbabilityColor(probability: number): string {
        // Clamp probability between 0 and 1
        const p = Math.max(0, Math.min(1, probability));
        // Linear interpolation from white (255) to medium blue (100)
        const whiteAmount = Math.round(255 - (255 - 100) * p);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    }

    function syncScroll(event: Event) {
        if ($isScrolling) return;

        const target = event.target as HTMLElement;
        const scrollLeft = target.scrollLeft;

        $isScrolling = true;

        const containers = document.querySelectorAll(".logits-display-container");
        for (const container of containers) {
            if (container !== target) {
                (container as HTMLElement).scrollLeft = scrollLeft;
            }
        }

        setTimeout(() => {
            $isScrolling = false;
        }, 10);
    }
</script>

<div class="logits-display-container {containerClass}" on:scroll={syncScroll}>
    <div class="logits-display">
        {#each tokenLogits as tokenPredictions, tokenIdx}
            <div class="token-predictions" class:has-ablation={hasAblation(tokenIdx)}>
                <div class="token-header">
                    <div class="token-name">
                        "{promptTokens[tokenIdx]}"
                    </div>
                </div>
                <div class="predictions-list">
                    {#each tokenPredictions as prediction}
                        <div
                            class="prediction-item"
                            style="background-color: {getProbabilityColor(prediction.probability)}"
                        >
                            <span class="prediction-token">"{prediction.token}"</span>
                            <span class="prediction-prob">{prediction.probability.toFixed(3)}</span>
                        </div>
                    {/each}
                </div>
            </div>
        {/each}
    </div>
</div>

<style>
    .logits-display-container {
        overflow-x: auto;
        margin-bottom: 0;
        border: 1px solid #ddd;
        border-radius: 6px;
        background: white;
        scrollbar-width: none;
    }

    .logits-display-container::-webkit-scrollbar {
        display: none;
    }

    .logits-display-container.original {
        border-color: #4caf50;
    }

    .logits-display {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 0;
        min-width: fit-content;
        padding: 0.5rem;
        width: max-content;
        overflow: visible;
    }

    .token-predictions {
        background: white;
        border-radius: 6px;
        border: 1px solid #ddd;
        width: 140px;
        padding: 0.25rem;
        margin-right: 0.25rem;
    }

    .token-header {
        color: #333;
        text-align: center;
        position: relative;
    }

    .token-predictions.has-ablation {
        border-color: #ff6b35;
        border-width: 2px;
        background: #fff8f5;
    }

    .ablation-indicator {
        position: absolute;
        top: -8px;
        right: -8px;
        font-size: 0.6rem;
        background: white;
        border-radius: 50%;
        padding: 2px;
    }

    .token-name {
        font-size: 0.8rem;
        color: #333;
        font-family: monospace;
        margin-top: 0.2rem;
        word-break: break-all;
    }

    .predictions-list {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
    }

    .prediction-item {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        border-radius: 2px;
        font-size: 0.7rem;
        padding: 0.1rem 0.2rem;
        margin: 0.05rem 0;
    }

    .prediction-token {
        font-family: monospace;
        font-size: 0.7rem;
        color: #333;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        text-align: left;
        flex: 1;
        margin-right: 0.2rem;
        padding: 0.1rem 0.3rem;
        border-radius: 2px;
    }

    .prediction-prob {
        font-family: monospace;
        font-size: 0.65rem;
        text-align: right;
        flex-shrink: 0;
    }
</style>