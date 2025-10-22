<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type { Status } from "$lib/api";
    import * as api from "$lib/api";
    import { onMount } from "svelte";

    import ActivationContextsTab from "$lib/components/ActivationContextsTab.svelte";
    // import InterventionsTab from "$lib/components/InterventionsTab.svelte";
    import { getWandbRunId } from "$lib";

    let loadingTrainRun: boolean = false;

    /** can be a wandb run path, or id. we should sanitse this */
    let trainWandbRunEntry: string | null = null;

    let status: Status = { train_run: null };

    async function loadStatus() {
        if (loadingTrainRun) return;

        console.log("getting status");

        try {
            status = await api.getStatus();
            loadingTrainRun = false;

            console.log("status:", status);
            if (!status.train_run) return;
            trainWandbRunEntry = status.train_run.wandb_path.split("/").pop()!;
        } catch (error) {
            console.error("error loading status", error);
        }
    }

    async function loadRun() {
        const input = trainWandbRunEntry?.trim();
        if (!input) return;

        try {
            loadingTrainRun = true;
            status = { train_run: null };
            trainWandbRunEntry = getWandbRunId(input);
            console.log("loading run", trainWandbRunEntry);
            await api.loadRun(trainWandbRunEntry);
            await loadStatus();
        } catch (error) {
            console.error("error loading run", error);
        } finally {
            loadingTrainRun = false;
        }
    }

    onMount(() => {
        loadStatus();
        setInterval(loadStatus, 5000);
    });

    let activeTab: "activation-contexts" | null = null;
</script>

<div class="app-layout">
    <aside class="sidebar">
        <div class="run-selector">
            <label for="wandb-run-id">W&B Run ID</label>
            <form on:submit|preventDefault={loadRun} class="input-group">
                <input
                    type="text"
                    id="wandb-run-id"
                    list="run-options"
                    bind:value={trainWandbRunEntry}
                    disabled={loadingTrainRun}
                    placeholder="Select or enter run ID"
                />
                <button type="submit" disabled={loadingTrainRun || !trainWandbRunEntry?.trim()}>
                    {loadingTrainRun ? "Loading..." : "Load Run"}
                </button>
            </form>
        </div>
        <div class="tab-navigation">
            {#if status.train_run}
                <button
                    class="tab-button"
                    class:active={activeTab === "activation-contexts"}
                    on:click={() => (activeTab = "activation-contexts")}
                >
                    Activation Contexts
                </button>
            {/if}
        </div>
        {#if status.train_run}
            <div class="config">
                <h4>Config</h4>
                <pre>{JSON.stringify(status.train_run?.config, null, 2)}</pre>
            </div>
        {/if}
    </aside>

    <!-- Main Content -->
    <div class="main-content">
        {#if status.train_run}
            <div class:hidden={activeTab !== "activation-contexts"}>
                <ActivationContextsTab />
            </div>
        {/if}
    </div>
</div>

<style>
    .app-layout {
        display: flex;
        min-height: 100vh;
    }

    .sidebar {
        background: #f8f9fa;
        border-right: 1px solid #dee2e6;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
    }

    .main-content {
        flex: 1;
        min-width: 0;
        padding: 2rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .tab-navigation {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .tab-button {
        padding: 0.75rem 1rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 500;
        color: #495057;
        transition: all 0.15s ease;
        text-align: left;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .tab-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }

    .tab-button:hover {
        color: #007bff;
        background: #f8f9fa;
        border-color: #007bff;
    }

    .tab-button.active {
        color: white;
        background: #007bff;
        border-color: #007bff;
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
    }

    .cluster-settings {
        padding: 1.25rem;
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }

    .cluster-settings h4 {
        margin: 0 0 1rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #343a40;
    }

    .cluster-settings form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .cluster-settings label {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #495057;
    }

    .cluster-settings select,
    .cluster-settings input {
        padding: 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 0.9rem;
        background: white;
    }

    .cluster-settings select:focus,
    .cluster-settings input:focus {
        outline: none;
        border-color: #007bff;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
    }

    .cluster-load {
        padding: 0.625rem 1rem;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .cluster-load.done {
        background: #28a745;
    }

    .cluster-load:hover:not(:disabled) {
        background: #0056b3;
    }

    .cluster-load:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }

    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #007bff;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

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

    .config {
        margin-top: 1rem;
    }

    .config h4 {
        margin: 0;
    }

    .config pre {
        margin: 0;
        font-size: 0.8rem;
    }
</style>
