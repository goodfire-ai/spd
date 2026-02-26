<script lang="ts">
    import type { CISnapshot } from "../../lib/promptAttributionsTypes";

    type Props = {
        snapshot: CISnapshot;
    };

    let { snapshot }: Props = $props();

    let canvas: HTMLCanvasElement | undefined = $state();

    const LABEL_WIDTH = 64;
    const HEADER_HEIGHT = 14;
    const CELL_MIN = 4;
    const CELL_MAX = 14;
    const MAX_GRID_WIDTH = 600;
    const MAX_GRID_HEIGHT = 320;

    function abbreviateLayer(layer: string): string {
        // "h.0.attn.q_proj" -> "0.q", "h.1.mlp.down_proj" -> "1.down"
        const m = layer.match(/(\d+)\.\w+\.(\w+)/);
        if (!m) return layer;
        const shortNames: Record<string, string> = {
            q_proj: "q",
            k_proj: "k",
            v_proj: "v",
            o_proj: "o",
            up_proj: "up",
            gate_proj: "gate",
            down_proj: "down",
            c_fc: "up",
            c_proj: "down",
        };
        return `${m[1]}.${shortNames[m[2]] ?? m[2]}`;
    }

    const nLayers = $derived(snapshot.layers.length);
    const seqLen = $derived(snapshot.seq_len);

    const cellSize = $derived(
        Math.max(
            CELL_MIN,
            Math.min(CELL_MAX, Math.floor(MAX_GRID_WIDTH / seqLen), Math.floor(MAX_GRID_HEIGHT / nLayers)),
        ),
    );
    const gridWidth = $derived(cellSize * seqLen);
    const gridHeight = $derived(cellSize * nLayers);
    const totalWidth = $derived(LABEL_WIDTH + gridWidth);
    const totalHeight = $derived(HEADER_HEIGHT + gridHeight);

    $effect(() => {
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = totalWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.width = `${totalWidth}px`;
        canvas.style.height = `${totalHeight}px`;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Clear
        ctx.clearRect(0, 0, totalWidth, totalHeight);

        // Draw layer labels
        ctx.font = `${Math.min(10, cellSize)}px "SF Mono", monospace`;
        ctx.fillStyle = "#b4b4b4";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let i = 0; i < nLayers; i++) {
            const y = HEADER_HEIGHT + i * cellSize + cellSize / 2;
            ctx.fillText(abbreviateLayer(snapshot.layers[i]), LABEL_WIDTH - 4, y);
        }

        // Draw grid cells
        for (let row = 0; row < nLayers; row++) {
            for (let col = 0; col < seqLen; col++) {
                const initial = snapshot.initial_alive[row][col];
                const current = snapshot.current_alive[row][col];
                const fraction = initial > 0 ? current / initial : 0;

                const x = LABEL_WIDTH + col * cellSize;
                const y = HEADER_HEIGHT + row * cellSize;

                // Color: accent-primary at full opacity for fraction=1, fading to bg-inset for fraction=0
                // Using accent-primary RGB: #7C4D33 = rgb(124, 77, 51)
                const alpha = fraction;
                if (initial === 0) {
                    // No components at this position — mark as empty
                    ctx.fillStyle = "#f0efeb";
                } else {
                    ctx.fillStyle = `rgba(124, 77, 51, ${Math.max(0.04, alpha)})`;
                }
                ctx.fillRect(x, y, cellSize - 0.5, cellSize - 0.5);
            }
        }
    });

    const initialL0 = $derived(snapshot.initial_alive.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0));
    const fractionRemaining = $derived(initialL0 > 0 ? snapshot.l0_total / initialL0 : 0);
</script>

<div class="optimization-grid">
    <div class="grid-header">
        <span class="step-label">
            Step {snapshot.step}/{snapshot.total_steps}
        </span>
        <span class="l0-label">
            L0: {Math.round(snapshot.l0_total)} / {initialL0}
            ({(fractionRemaining * 100).toFixed(0)}%)
        </span>
        {#if snapshot.loss > 0}
            <span class="loss-label">loss: {snapshot.loss.toFixed(4)}</span>
        {/if}
    </div>
    <canvas bind:this={canvas}></canvas>
</div>

<style>
    .optimization-grid {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--space-2);
    }

    .grid-header {
        display: flex;
        gap: var(--space-4);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .l0-label {
        color: var(--accent-primary);
        font-weight: 600;
    }

    canvas {
        image-rendering: pixelated;
    }
</style>
