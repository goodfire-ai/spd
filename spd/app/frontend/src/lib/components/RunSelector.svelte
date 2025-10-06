<script lang="ts">
    import * as api from "$lib/api";

    export let trainWandbRunId: string | null;
    export let loadingRun: boolean;

    async function loadRun() {
        if (!trainWandbRunId?.trim()) return;

        loadingRun = true;
        await api.loadRun(trainWandbRunId);
        loadingRun = false;
    }
</script>

<div class="run-selector">
    <label for="wandb-run-id">W&B Run ID</label>
    <div class="input-group">
        <input
            type="text"
            id="wandb-run-id"
            list="run-options"
            bind:value={trainWandbRunId}
            disabled={loadingRun}
            placeholder="Select or enter run ID"
        />
        <button on:click={loadRun} disabled={loadingRun || !trainWandbRunId?.trim()}>
            {loadingRun ? "Loading..." : "Load Run"}
        </button>
    </div>
</div>

<style>
    .run-selector {
        margin-bottom: 1rem;
    }

    .run-selector label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #333;
        font-size: 0.9rem;
    }

    .input-group {
        display: flex;
        gap: 0.5rem;
    }

    .input-group input[type="text"] {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1rem;
    }

    .input-group input[type="text"]:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }

    .input-group button {
        padding: 0.5rem 1rem;
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 1rem;
        cursor: pointer;
        white-space: nowrap;
    }

    .input-group button:hover:not(:disabled) {
        background-color: #357abd;
    }

    .input-group button:disabled {
        background-color: #ccc;
        cursor: not-allowed;
    }
</style>
