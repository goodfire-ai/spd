<script lang="ts">
    import type { Status } from "./lib/api";
    import * as api from "./lib/api";

    import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
    import { parseWandbRunPath } from "./lib";

    let loadingTrainRun = $state<boolean | null>(false);

    /** can be a wandb run path, or id. we sanitize this on sumbit */
    let trainWandbRunEntry = $state<string | null>(null);

    let status = $state<Status>({ train_run: null });
    let backendError = $state<string | null>(null);

    async function loadStatus() {
        if (loadingTrainRun) return;
        try {
            const newStatus = await api.getStatus();
            loadingTrainRun = false;

            // if we have a run, but the backend says null, then we're out of sync (backend likely restarted)
            // in this case, we keep the local state so the user can still see what they were looking at
            if (status.train_run && !newStatus.train_run) {
                backendError = "Backend state lost (restarted). Showing cached view.";
                return;
            }

            // otherwise, we update the status
            status = newStatus;
            backendError = null;

            if (!status.train_run) return;
            trainWandbRunEntry = status.train_run.wandb_path;
        } catch (error) {
            console.error("error loading status", error);
            // if the backend is down, we keep the local state
            backendError = "Backend unreachable. Showing cached view.";
        }
    }

    async function loadRun(event: Event) {
        event.preventDefault();
        const input = trainWandbRunEntry?.trim();
        if (!input) return;
        try {
            loadingTrainRun = true;
            status = { train_run: null };
            const wandbRunPath = parseWandbRunPath(input);
            console.log("loading run", wandbRunPath);
            trainWandbRunEntry = wandbRunPath;
            await api.loadRun(wandbRunPath);
            await loadStatus();
        } catch (error) {
            console.error("error loading run", error);
        } finally {
            loadingTrainRun = false;
        }
    }

    //   when the page loads, and every 5 seconds thereafter, load the status
    $effect(() => {
        loadStatus();
        const interval = setInterval(loadStatus, 5000);
        // return cleanup function to stop the polling
        return () => clearInterval(interval);
    });

    let activeTab = $state<"activation-contexts" | null>(null);
</script>

<div class="app-layout">
    <aside class="sidebar">
        <div class="section-heading">W&B Run ID</div>
        <form onsubmit={loadRun} class="input-group">
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
        <div class="tab-navigation">
            {#if status.train_run}
                <button
                    class="tab-button"
                    class:active={activeTab === "activation-contexts"}
                    onclick={() => {
                        console.log("clicking activation contexts tab");
                        activeTab = "activation-contexts";
                    }}
                >
                    Activation Contexts
                </button>
            {/if}
        </div>
        {#if status.train_run}
            <div class="config">
                <div class="section-heading">Config</div>
                <pre>{status.train_run?.config_yaml}</pre>
            </div>
        {/if}
    </aside>

    <div class="main-content">
        {#if backendError}
            <div class="warning-banner">
                ⚠️ {backendError}
            </div>
        {/if}
        {#if status.train_run && activeTab === "activation-contexts"}
            <ActivationContextsTab />
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
        gap: 0.5rem;
        flex-direction: column;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
    }

    .main-content {
        flex: 1;
        min-width: 0;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .warning-banner {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem 1rem;
        border: 1px solid #ffeeba;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .tab-navigation {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .tab-button {
        padding: 0.75rem 0.5rem;
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

    .section-heading {
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
        padding: 0.5rem 0.5rem;
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
        margin-top: 0.5rem;
    }

    .config pre {
        margin: 0;
        font-size: 0.8rem;
    }
</style>
