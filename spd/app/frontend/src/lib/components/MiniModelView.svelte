<script lang="ts">
    type ComponentInfo = { module: string; index: number };
    export let components: ComponentInfo[];
    if (components == null) throw new Error('Components are required');

    export let layerCount: number;

    const SUBLAYER_COLUMNS = [
        'self_attn.q_proj',
        'self_attn.k_proj',
        'self_attn.v_proj',
        'self_attn.o_proj',
        'mlp.gate_proj',
        'mlp.up_proj',
        'mlp.down_proj'
    ];

    function getLayer(moduleName: string): number {
        const parts = moduleName.split('.');
        const i = parts.indexOf('layers');
        if (i >= 0 && i + 1 < parts.length) {
            const n = parseInt(parts[i + 1]);
            return Number.isFinite(n) ? n : 0;
        }
        return 0;
    }

    function getColumnIndex(moduleName: string): number {
        for (let i = 0; i < SUBLAYER_COLUMNS.length; i++) {
            if (moduleName.includes(SUBLAYER_COLUMNS[i])) return i;
        }
        return -1; // ignore others
    }

    type Cell = { count: number; indices: number[] };
    $: grid = Array.from({ length: layerCount }, () =>
        SUBLAYER_COLUMNS.map(() => ({ count: 0, indices: [] as number[] }))
    );

    $: {
        // reset
        for (let r = 0; r < layerCount; r++) {
            for (let c = 0; c < SUBLAYER_COLUMNS.length; c++) {
                grid[r][c].count = 0;
                grid[r][c].indices = [];
            }
        }
        // accumulate
        for (const comp of components) {
            const layer = getLayer(comp.module);
            const col = getColumnIndex(comp.module);
            if (layer >= 0 && layer < layerCount && col >= 0) {
                grid[layer][col].count += 1;
                grid[layer][col].indices.push(comp.index);
            }
        }
    }

    $: maxCount = Math.max(1, ...grid.flat().map((c) => c.count));

    function colorFor(v: number): string {
        if (v <= 0) return '#eef2f6';
        const t = Math.min(1, v / maxCount);
        const r = 230 - Math.round(150 * t);
        const g = 240 - Math.round(170 * t);
        const b = 255;
        return `rgb(${r},${g},${b})`;
    }

    // tooltip state
    let tipVisible = false;
    let tipX = 0;
    let tipY = 0;
    let tipText = '';

    function showTip(e: MouseEvent, layer: number, col: number, cell: Cell) {
        tipVisible = true;
        tipX = (e as MouseEvent).clientX + 10;
        tipY = (e as MouseEvent).clientY + 10;
        const modulePath = SUBLAYER_COLUMNS[col];
        const moduleName = `model.layers.${layer}.${modulePath}`;
        const indices = cell.indices.join(',');
        tipText = `${moduleName}\nComponents: ${cell.count}\nIndices: ${indices || 'none'}`;
    }
    function moveTip(e: MouseEvent) {
        tipX = (e as MouseEvent).clientX + 10;
        tipY = (e as MouseEvent).clientY + 10;
    }
    function hideTip() {
        tipVisible = false;
    }
</script>

<div class="mini-model-view" style={`--cols:${SUBLAYER_COLUMNS.length}; --rows:${layerCount};`}>
    {#each Array.from({ length: layerCount }) as _, r}
        {#each SUBLAYER_COLUMNS as path, c}
            <div
                class="cell"
                role="img"
                aria-label={`Layer ${r}, ${path} - ${grid[r][c].count} components`}
                style={`background:${colorFor(grid[r][c].count)}`}
                on:mouseenter={(e) => showTip(e, r, c, grid[r][c])}
                on:mousemove={moveTip}
                on:mouseleave={hideTip}
            ></div>
        {/each}
    {/each}
    {#if tipVisible}
        <div class="tooltip" style={`left:${tipX}px; top:${tipY}px;`}>{tipText}</div>
    {/if}
</div>

<style>
    .mini-model-view {
        position: relative;
        display: grid;
        grid-template-columns: repeat(var(--cols), 12px);
        grid-template-rows: repeat(var(--rows), 12px);
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
    .tooltip {
        position: fixed;
        z-index: 1000;
        pointer-events: none;
        background: rgba(0,0,0,0.9);
        color: #fff;
        font-size: 11px;
        line-height: 1.2;
        padding: 6px 8px;
        border-radius: 4px;
        white-space: pre-line;
    }
</style>

