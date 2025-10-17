<script lang="ts">
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
            return "#f0f0f0";
        }

        // Clamp between 0 and 1
        const v = Math.max(0, Math.min(1, value));

        // Linear gradient: white (0) -> blue (1)
        const whiteAmount = Math.round(255 * (1 - v));
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    };
</script>

<div class="heatmap-wrapper">
    <div class="heatmap">
        {#each data as row, i}
            <div class="heatmap-row">
                {#each row as value, j}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <!-- svelte-ignore a11y_click_events_have_key_events -->
                    <div
                        class="heatmap-cell"
                        class:disabled={isDisabled(i, j)}
                        style="background-color: {getHeatmapColor(value, i, j)};"
                        title="Components {indices[i]} × {indices[j]}: {value.toFixed(
                            3
                        )}{isDisabled(i, j) ? ' (disabled)' : ''}"
                    ></div>
                {/each}
            </div>
        {/each}
    </div>
</div>

<style>
    .heatmap-wrapper {
        width: 100%;
        aspect-ratio: 1;
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }

    .heatmap {
        display: flex;
        gap: 0;
        flex-direction: column;
        width: 100%;
        height: 100%;
    }

    .heatmap-row {
        display: flex;
        flex: 1;
    }

    .heatmap-cell {
        flex: 1;
    }
</style>
