<script lang="ts">
    type Props = {
        topK: number;
        nodeLayout: "importance" | "shuffled" | "jittered";
        filteredEdgeCount: number | null;
        onTopKChange: (value: number) => void;
        onLayoutChange: (value: "importance" | "shuffled" | "jittered") => void;
    };

    let { topK, nodeLayout, filteredEdgeCount, onTopKChange, onLayoutChange }: Props = $props();
</script>

<div class="controls-bar">
    <span class="controls-label">View</span>
    <label>
        <span>Top K</span>
        <input
            type="number"
            value={topK}
            oninput={(e) => onTopKChange(parseInt(e.currentTarget.value) || 800)}
            min={10}
            max={10000}
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

    .controls-label {
        font-size: var(--text-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-family: var(--font-sans);
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
        text-transform: uppercase;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    input[type="number"] {
        width: 60px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-warm-dim);
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
        border-color: var(--accent-warm-dim);
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
        text-transform: uppercase;
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
