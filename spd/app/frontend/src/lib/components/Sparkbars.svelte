<script lang="ts">
    export let bins: number[] = [];
    export let width = 200;
    export let height = 60;
    export let color = "#4169E1";
    export let logScale = true;
    export let margin = 2;

    $: maxVal = bins.length ? Math.max(...bins) : 0;
    $: barCount = bins.length;
    $: innerW = Math.max(0, width - margin * 2);
    $: innerH = Math.max(0, height - margin * 2);
    $: barW = barCount > 0 ? innerW / barCount : 0;

    function yFor(v: number): number {
        if (maxVal <= 0) return innerH;
        if (logScale) {
            const lv = Math.log10(Math.max(1, v));
            const maxLv = Math.log10(Math.max(1, maxVal));
            const ratio = maxLv === 0 ? 0 : lv / maxLv;
            return innerH * (1 - ratio);
        }
        return innerH * (1 - v / maxVal);
    }
</script>

<svg {width} {height} class="sparkbars">
    <!-- background -->
    <rect x="0" y="0" width={width} height={height} fill="transparent" />
    <g transform={`translate(${margin}, ${margin})`}>
        {#each bins as v, i}
            {#if barW >= 0.5}
                <rect
                    x={i * barW}
                    y={yFor(v)}
                    width={Math.max(0, barW - 1)}
                    height={innerH - yFor(v)}
                    fill={color}
                />
            {/if}
        {/each}
    </g>
    
</svg>

<style>
    .sparkbars {
        display: block;
    }
</style>

