<script lang="ts">
    import type { LayerCIs, SparseVector } from "$lib/api";
    import {
        ablationSubcomponentMask,
        multiSelectMode,
        selectedTokensForCombining
    } from "$lib/stores/componentState";
    import { createEventDispatcher } from "svelte";
    import MaskCombinePanel from "./MaskCombinePanel.svelte";

    const dispatch = createEventDispatcher();

    export let result: { layer_cis: LayerCIs[]; prompt_tokens: string[] };
    export let promptId: string;
    export let onCellClick: (
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        tokenCis: SparseVector
    ) => void;

    function handleMaskCreated() {
        dispatch("maskCreated");
    }

    function handleCellClick(
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        tokenCis: SparseVector
    ) {
        if ($multiSelectMode) {
            const existingIndex = $selectedTokensForCombining.findIndex(
                (t) => t.layer === layer && t.tokenIdx === tokenIdx
            );

            if (existingIndex >= 0) {
                // Remove if already selected
                $selectedTokensForCombining = $selectedTokensForCombining.filter(
                    (_, idx) => idx !== existingIndex
                );
            } else {
                // Add to selection
                $selectedTokensForCombining = [
                    ...$selectedTokensForCombining,
                    { layer, tokenIdx, token }
                ];
            }
        } else {
            // Normal click behavior - open popup
            onCellClick(token, tokenIdx, layer, layerIdx, tokenCis);
        }
    }

    function isTokenSelected(layer: string, tokenIdx: number): boolean {
        return $selectedTokensForCombining.some(
            (t) => t.layer === layer && t.tokenIdx === tokenIdx
        );
    }

    let globalMax = Math.max(
        ...result.layer_cis.flatMap((layer) =>
            layer.token_cis.map((tokenCIs) => tokenCIs.subcomponent_cis.l0)
        )
    );

    $: layer_cis = result.layer_cis.toReversed();

    // Make this reactive so it updates when $runAblation changes
    $: getColorFroml0 = (l0: number, layerName: string, tokenIdx: number): string => {
        const intensity = Math.max(0, Math.min(1, l0 / globalMax));
        const disabledComponents = $ablationSubcomponentMask[layerName]?.[tokenIdx]?.length ?? 0;
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
    <MaskCombinePanel {promptId} on:maskCreated={handleMaskCreated} />
    <div class="heatmap-container-horiz">
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
                                class:selected={isTokenSelected(layer.module, tokenIdx)}
                                class:multi-select-mode={$multiSelectMode}
                                style="background: {getColorFroml0(
                                    layer.token_cis[tokenIdx].subcomponent_cis.l0,
                                    layer.module,
                                    tokenIdx
                                )}"
                                title="L0={layer.token_cis[tokenIdx].subcomponent_cis.l0}"
                                on:click={() =>
                                    handleCellClick(
                                        token,
                                        tokenIdx,
                                        layer.module,
                                        layerIdx,
                                        layer.token_cis[tokenIdx].subcomponent_cis
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
</div>

<style>
    .heatmap-container {
        display: flex;
        flex-direction: column;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1rem;
        background-color: #fafafa;
    }

    .heatmap-container-horiz {
        flex: 1;
        display: flex;
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

    /* Multi-select mode styling */
    .heatmap-cell.multi-select-mode {
        cursor: pointer;
    }

    .heatmap-cell.multi-select-mode:hover {
        border: 2px solid #4caf50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }

    /* Selected cell styling */
    .heatmap-cell.selected {
        border: 3px solid #4caf50 !important;
        box-shadow: 0 0 8px rgba(76, 175, 80, 0.7);
        position: relative;
        z-index: 5;
    }

    .heatmap-cell.selected::after {
        content: "✓";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 14px;
        font-weight: bold;
        text-shadow: 0 0 3px rgba(0, 0, 0, 0.7);
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
