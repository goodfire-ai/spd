<script lang="ts">
    export let title: string;
    export let data: number[][];
    export let indices: number[];
    export let disabledIndices: number[] = [];

    function isDisabled(i: number, j: number): boolean {
        // Check if either row or column corresponds to a disabled component
        return disabledIndices.includes(indices[i]) || disabledIndices.includes(indices[j]);
    }

    // Make color calculation reactive to disabledIndices changes
    $: getHeatmapColor = (value: number, i: number, j: number): string => {
        // If disabled, return soft gray
        if (disabledIndices.includes(indices[i]) || disabledIndices.includes(indices[j])) {
            return '#f0f0f0';
        }

        // Clamp between 0 and 1
        const v = Math.max(0, Math.min(1, value));

        // Linear gradient: white (0) -> blue (1)
        const whiteAmount = Math.round(255 * (1 - v));
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    };
</script>

<div class="similarity-plot">
    <h4>{title}</h4>
    <div class="heatmap-wrapper">
        <div class="heatmap">
            <div class="axis-labels-y">
                {#each indices as idx}
                    <div class="label">{idx}</div>
                {/each}
            </div>
            <div class="heatmap-content">
                {#each data as row, i}
                    <div class="heatmap-row">
                        {#each row as value, j}
                            <!-- svelte-ignore a11y_no_static_element_interactions -->
                            <!-- svelte-ignore a11y_click_events_have_key_events -->
                            <div
                                class="heatmap-cell"
                                class:disabled={isDisabled(i, j)}
                                style="background-color: {getHeatmapColor(value, i, j)}"
                                title="Components {indices[i]} × {indices[j]}: {value.toFixed(3)}{isDisabled(i, j) ? ' (disabled)' : ''}"
                            >
                            </div>
                        {/each}
                    </div>
                {/each}
            </div>
        </div>
    </div>
</div>

<style>
    .similarity-plot {
        margin: 1rem 0;
    }

    .similarity-plot h4 {
        margin: 0 0 0.5rem 0;
        color: #333;
        font-size: 0.9rem;
    }

    .heatmap-wrapper {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }

    .heatmap {
        display: flex;
        gap: 0.25rem;
    }

    .axis-labels-y {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        padding-right: 0.25rem;
    }

    .label {
        font-size: 0.7rem;
        color: #666;
        text-align: center;
        min-width: 1.2em;
        height: 20px;
    }

    .heatmap-content {
        display: flex;
        flex-direction: column;
    }

    .heatmap-row {
        display: flex;
    }

    .heatmap-cell {
        width: 20px;
        height: 20px;
        border: 0.5px solid rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: transform 0.1s;
    }

    .heatmap-cell:hover {
        transform: scale(1.1);
        z-index: 10;
        border: 1px solid #333;
    }

    .heatmap-cell.disabled {
        cursor: not-allowed;
    }

    .heatmap-cell.disabled:hover {
        transform: none;
        border: 0.5px solid rgba(0, 0, 0, 0.1);
    }
</style>