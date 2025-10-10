<script lang="ts">
    import type { AblationStats } from "$lib/api";

    export let ablationStats: AblationStats;
    export let promptTokens: string[];

    // Find global max for color scaling
    let globalMaxActive = Math.max(
        ...ablationStats.layer_stats.flatMap((layer) =>
            layer.token_stats.map((stat) => stat.original_active_count)
        )
    );

    function getColorFromCounts(originalActive: number, ablated: number): string {
        const intensity = Math.max(0, Math.min(1, originalActive / globalMaxActive));
        const ablatedRatio = originalActive > 0 ? ablated / originalActive : 0;

        // Blue depth for active components
        const whiteAmount = Math.round((1 - intensity) * 255);
        const baseColor = `rgb(${whiteAmount}, ${whiteAmount}, 255)`;

        if (ablatedRatio === 0) {
            return baseColor;
        }

        // Red bar for ablated components
        const ablatedPercent = Math.round(ablatedRatio * 100);
        return `linear-gradient(to right, #ff4444 0%, #ff4444 ${ablatedPercent}%, ${baseColor} ${ablatedPercent}%, ${baseColor} 100%)`;
    }

    $: layer_stats = ablationStats.layer_stats.toReversed();
</script>

<div class="ablation-heatmap">
    <h4>Ablation Impact</h4>
    <div class="heatmap-container-horiz">
        <div class="layer-labels">
            <div class="layer-label-spacer"></div>
            {#each layer_stats as layer}
                <div class="layer-label">{layer.module}</div>
            {/each}
        </div>

        <div class="heatmap-scroll-area">
            <div class="heatmap-grid">
                {#each layer_stats as layer}
                    <div class="heatmap-row">
                        {#each layer.token_stats as tokenStat, tokenIdx}
                            <div
                                class="heatmap-cell"
                                style="background: {getColorFromCounts(
                                    tokenStat.original_active_count,
                                    tokenStat.ablated_count
                                )}"
                                title="Active: {tokenStat.original_active_count}, Ablated: {tokenStat.ablated_count}, Magnitude: {tokenStat.ablated_magnitude.toFixed(
                                    3
                                )}"
                            ></div>
                        {/each}
                    </div>
                {/each}

                <div class="token-labels">
                    {#each promptTokens as token}
                        <div class="token-label">{token}</div>
                    {/each}
                </div>
            </div>
        </div>
    </div>
    <div class="legend">
        <span><strong>Blue depth:</strong> # active components</span>
        <span><strong>Red bar:</strong> # ablated components</span>
    </div>
</div>

<style>
    .ablation-heatmap {
        margin-top: 1rem;
        padding: 0.75rem;
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
    }

    .ablation-heatmap h4 {
        margin: 0 0 0.5rem 0;
        color: #333;
        font-size: 0.95rem;
    }

    .heatmap-container-horiz {
        display: flex;
    }

    .layer-labels {
        display: flex;
        flex-direction: column;
        margin-right: 0.5rem;
        min-width: 120px;
    }

    .layer-label-spacer {
        height: 0px;
    }

    .layer-label {
        display: flex;
        align-items: center;
        height: 24px;
        padding: 0.25rem 0.5rem;
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #555;
        margin-bottom: 2px;
    }

    .heatmap-scroll-area {
        flex: 1;
        overflow-x: auto;
        overflow-y: hidden;
    }

    .heatmap-grid {
        display: flex;
        flex-direction: column;
    }

    .heatmap-row {
        display: flex;
        gap: 2px;
        margin-bottom: 2px;
    }

    .heatmap-cell {
        width: 20px;
        height: 24px;
        border: 1px solid #ccc;
        border-radius: 3px;
        flex-shrink: 0;
        cursor: default;
    }

    .token-labels {
        display: flex;
        gap: 2px;
        margin-top: 4px;
    }

    .token-label {
        width: 20px;
        font-size: 0.65rem;
        color: #666;
        text-align: center;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .legend {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #666;
    }
</style>
