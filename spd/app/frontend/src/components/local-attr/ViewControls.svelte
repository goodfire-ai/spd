<script lang="ts">
    type Props = {
        topK: number;
        nodeLayout: "importance" | "shuffled" | "jittered";
        onTopKChange: (value: number) => void;
        onLayoutChange: (value: "importance" | "shuffled" | "jittered") => void;
    };

    let { topK, nodeLayout, onTopKChange, onLayoutChange }: Props = $props();
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
</div>

<style>
    .controls-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    .controls-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #9e9e9e;
    }

    label {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: #616161;
    }

    label span {
        font-weight: 500;
    }

    input[type="number"] {
        width: 60px;
        padding: 0.25rem 0.4rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    select {
        padding: 0.25rem 0.4rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
