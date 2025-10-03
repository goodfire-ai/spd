<script lang="ts">
    type ComponentInfo = { module: string; index: number };
    export let components: ComponentInfo[] = [];
    export let maxLayers: number | null = null;

    function getLayer(moduleName: string): number {
        const parts = moduleName.split('.');
        const i = parts.indexOf('layers');
        if (i >= 0 && i + 1 < parts.length) {
            const n = parseInt(parts[i + 1]);
            return Number.isFinite(n) ? n : 0;
        }
        return 0;
    }

    function getType(moduleName: string): 'attn' | 'mlp' | 'other' {
        if (moduleName.includes('self_attn')) return 'attn';
        if (moduleName.includes('mlp')) return 'mlp';
        return 'other';
    }

    $: layerCount = (() => {
        const max = components.reduce((m, c) => Math.max(m, getLayer(c.module)), 0);
        return (maxLayers ?? max) + 1;
    })();

    $: attnCounts = Array.from({ length: layerCount }).map(() => 0);
    $: mlpCounts = Array.from({ length: layerCount }).map(() => 0);

    $: {
        // accumulate counts
        attnCounts.fill(0);
        mlpCounts.fill(0);
        for (const c of components) {
            const layer = getLayer(c.module);
            const type = getType(c.module);
            if (layer >= 0 && layer < layerCount) {
                if (type === 'attn') attnCounts[layer] += 1;
                else if (type === 'mlp') mlpCounts[layer] += 1;
            }
        }
    }

    $: maxCount = Math.max(1, ...attnCounts, ...mlpCounts);

    function colorFor(v: number): string {
        // simple blue scale
        const t = Math.min(1, v / maxCount);
        const r = 230 - Math.round(150 * t);
        const g = 240 - Math.round(170 * t);
        const b = 255;
        return `rgb(${r},${g},${b})`;
    }
</script>

<div class="mini-model-view" style={`--cols:${layerCount};`}>
    {#each Array.from({ length: layerCount }) as _, i}
        <div class="cell" title={`L${i} attn: ${attnCounts[i]}`} style={`background:${colorFor(attnCounts[i])}`}></div>
    {/each}
    {#each Array.from({ length: layerCount }) as _, i}
        <div class="cell" title={`L${i} mlp: ${mlpCounts[i]}`} style={`background:${colorFor(mlpCounts[i])}`}></div>
    {/each}
</div>

<style>
    .mini-model-view {
        display: grid;
        grid-template-columns: repeat(var(--cols), 12px);
        grid-auto-rows: 12px;
        gap: 2px;
        padding: 2px;
        background: #fff;
        border: 1px solid #eee;
        border-radius: 3px;
        width: max-content;
    }
    .cell {
        width: 12px;
        height: 12px;
        border-radius: 2px;
    }
</style>

