<script lang="ts">
    import type { NormalizeType } from "../../lib/localAttributionsApi";

    type Props = {
        topK: number;
        nodeLayout: "importance" | "shuffled" | "jittered";
        componentGap: number;
        layerGap: number;
        filteredEdgeCount: number | null;
        normalizeEdges: NormalizeType;
        onTopKChange: (value: number) => void;
        onLayoutChange: (value: "importance" | "shuffled" | "jittered") => void;
        onComponentGapChange: (value: number) => void;
        onLayerGapChange: (value: number) => void;
        onNormalizeChange: (value: NormalizeType) => void;
    };

    let {
        topK,
        nodeLayout,
        componentGap,
        layerGap,
        filteredEdgeCount,
        normalizeEdges,
        onTopKChange,
        onLayoutChange,
        onComponentGapChange,
        onLayerGapChange,
        onNormalizeChange,
    }: Props = $props();
</script>

<div class="controls-bar">
    <label>
        <span>Norm</span>
        <select value={normalizeEdges} onchange={(e) => onNormalizeChange(e.currentTarget.value as NormalizeType)}>
            <option value="none">None</option>
            <option value="target">Target</option>
            <option value="layer">Layer</option>
        </select>
    </label>
    <label>
        <span>Top K</span>
        <input
            type="number"
            value={topK}
            oninput={(e) => {
                if (e.currentTarget.value === "") return;
                onTopKChange(parseInt(e.currentTarget.value));
            }}
            min={0}
            max={10_000}
            step={100}
        />
    </label>
    <label>
        <span>Layout</span>
        <select
            value={nodeLayout}
            onchange={(e) => onLayoutChange(e.currentTarget.value as "importance" | "shuffled" | "jittered")}
        >
            <option value="importance">Importance</option>
            <option value="shuffled">Shuffled</option>
            <option value="jittered">Jittered</option>
        </select>
    </label>
    <label>
        <span>Node Gap</span>
        <input
            type="number"
            value={componentGap}
            oninput={(e) => {
                if (e.currentTarget.value === "") return;
                onComponentGapChange(parseInt(e.currentTarget.value));
            }}
            min={0}
            max={20}
            step={1}
        />
    </label>
    <label>
        <span>Layer Gap</span>
        <input
            type="number"
            value={layerGap}
            oninput={(e) => {
                if (e.currentTarget.value === "") return;
                onLayerGapChange(parseInt(e.currentTarget.value));
            }}
            min={10}
            max={100}
            step={5}
        />
    </label>

    {#if filteredEdgeCount !== null}
        <div class="legend">
            <span class="edge-count">Showing {filteredEdgeCount} edges</span>
            <span class="legend-item">
                <span class="edge-pos"></span> Positive
            </span>
            <span class="legend-item">
                <span class="edge-neg"></span> Negative
            </span>
        </div>
    {/if}
</div>

<style>
    .controls-bar {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    label {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-sans);
    }

    label span {
        font-weight: 500;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    input[type="number"] {
        width: 75px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    select {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        cursor: pointer;
    }

    select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .legend {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        margin-left: auto;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
    }

    .edge-pos,
    .edge-neg {
        display: inline-block;
        width: 16px;
        height: 2px;
    }

    .edge-pos {
        background: var(--status-info);
    }

    .edge-neg {
        background: var(--status-negative);
    }

    .edge-count {
        font-weight: 500;
        color: var(--text-primary);
    }
</style>
