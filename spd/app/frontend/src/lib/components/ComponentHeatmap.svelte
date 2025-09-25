<script lang="ts">
    import type { LayerCIs } from "$lib/api";
    import { runAblation } from "$lib/stores/componentState";

    export let result: { layer_cis: LayerCIs[]; prompt_tokens: string[] };
    export let onCellClick: (
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        token_ci: any
    ) => void;

    let globalMax = Math.max(
        ...result.layer_cis.flatMap((layer) => layer.token_cis.map((tokenCIs) => tokenCIs.l0))
    );

    $: layer_cis = result.layer_cis.toReversed();

    // Make this reactive so it updates when $runAblation changes
    $: getColorFroml0 = (l0: number, layerName: string, tokenIdx: number): string => {
        const intensity = Math.max(0, Math.min(1, l0 / globalMax));
        const disabledComponents = $runAblation[layerName]?.[tokenIdx]?.length ?? 0;
        const totalComponents = l0;
        const disabledRatio = totalComponents > 0 ? disabledComponents / totalComponents : 0;

        const whiteAmount = Math.round((1 - intensity) * 255);
        const baseColor = `rgb(${whiteAmount}, ${whiteAmount}, 255)`;

        if (disabledRatio === 0) {
            return baseColor;
        }

        const disabledPercent = Math.round(disabledRatio * 100);
        return `linear-gradient(to right, #ff4444 0%, #ff4444 ${disabledPercent}%, ${baseColor} ${disabledPercent}%, ${baseColor} 100%)`;
    };
</script>

<div class="heatmap-container">
    <div class="layer-labels">
        <div class="layer-label-spacer"></div>
        {#each layer_cis as layer}
            <div class="layer-label">{layer.module}</div>
        {/each}
    </div>

    <div class="heatmap-scroll-area">
        <div class="heatmap-grid">
            {#each layer_cis as layer, layerIdx}
                <div class="heatmap-row">
                    {#each result.prompt_tokens as token, tokenIdx}
                        <!-- svelte-ignore a11y_click_events_have_key_events -->
                        <!-- svelte-ignore a11y_no_static_element_interactions -->
                        <div
                            class="heatmap-cell"
                            style="background: {getColorFroml0(
                                layer.token_cis[tokenIdx].l0,
                                layer.module,
                                tokenIdx
                            )}"
                            title="L0={layer.token_cis[tokenIdx].l0}"
                            on:click={() =>
                                onCellClick(
                                    token,
                                    tokenIdx,
                                    layer.module,
                                    layerIdx,
                                    layer.token_cis[tokenIdx]
                                )}
                        ></div>
                    {/each}
                </div>
            {/each}

            <div class="token-labels">
                {#each result.prompt_tokens as token}
                    <div class="token-label">{token}</div>
                {/each}
            </div>
        </div>
    </div>
</div>

<style>
    .heatmap-container {
        flex: 1;
        display: flex;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1rem;
        background-color: #fafafa;
    }

    .layer-labels {
        display: flex;
        flex-direction: column;
        margin-right: 0.5rem;
        flex-shrink: 0;
    }

    .layer-label-spacer {
        height: 40px;
        order: 999;
    }

    .layer-label {
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 0.5rem;
        font-size: 0.9rem;
        font-weight: bold;
        color: #555;
        min-width: 100px;
        margin-bottom: 2px;
    }

    .heatmap-scroll-area {
        flex: 1;
        overflow-x: auto;
    }

    .heatmap-grid {
        display: flex;
        flex-direction: column;
        min-width: fit-content;
    }

    .heatmap-row {
        display: flex;
        margin-bottom: 2px;
    }

    .heatmap-cell {
        width: 50px;
        height: 20px;
        border: 1px solid #fff;
        cursor: pointer;
        transition: transform 0.1s ease;
    }

    .heatmap-cell:hover {
        border: 2px solid #241d8c;
        z-index: 10;
        position: relative;
    }

    .token-labels {
        display: flex;
        height: 40px;
        margin-bottom: 2px;
    }

    .token-label {
        width: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: bold;
        color: #333;
        text-align: center;
        padding: 0 2px;
        word-break: break-all;
        border-right: 1px solid #eee;
    }
</style>