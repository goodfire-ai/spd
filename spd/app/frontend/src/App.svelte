<script lang="ts">
    import { RenderScan } from "svelte-render-scan";
    import type { LoadedRun } from "./lib/api";
    import * as api from "./lib/api";

    import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
    import LocalAttributionsTab from "./components/LocalAttributionsTab.svelte";
    import { parseWandbRunPath } from "./lib";

    let loadingTrainRun = $state(false);

    /** can be a wandb run path, or id. we sanitize this on sumbit */
    let trainWandbRunEntry = $state<string | null>(null);
    let contextLength = $state<number>(64);

    let loadedRun = $state<LoadedRun | null>(null);
    let backendError = $state<string | null>(null);

    async function loadStatus() {
        if (loadingTrainRun) return;
        try {
            const newLoadedRun = await api.getStatus();
            loadingTrainRun = false;

            // if we have a run, but the backend says null, then we're out of sync (backend likely restarted)
            // in this case, we keep the local state so the user can still see what they were looking at
            if (loadedRun && !newLoadedRun) {
                backendError = "Backend state lost (restarted). Showing cached view.";
                return;
            }

            // otherwise, we update the status
            loadedRun = newLoadedRun;
            backendError = null;

            if (!loadedRun) return;
            trainWandbRunEntry = loadedRun.wandb_path;
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
            loadedRun = null;
            const wandbRunPath = parseWandbRunPath(input);
            console.log("loading run", wandbRunPath);
            trainWandbRunEntry = wandbRunPath;
            await api.loadRun(wandbRunPath, contextLength);
            await loadStatus();
        } catch (error) {
            console.error("error loading run", error);
        } finally {
            loadingTrainRun = false;
        }
    }

    $effect(() => void loadStatus());

    let activeTab = $state<"activation-contexts" | "local-attributions" | null>(null);
    let showConfig = $state(false);
</script>

<RenderScan />
<div class="app-layout">
    <header class="top-bar">
        <form onsubmit={loadRun} class="run-input">
            <input
                type="text"
                id="wandb-run-id"
                list="run-options"
                bind:value={trainWandbRunEntry}
                disabled={loadingTrainRun}
                placeholder="W&B Run ID"
            />
            <input
                type="number"
                id="context-length"
                bind:value={contextLength}
                disabled={loadingTrainRun}
                min="1"
                max="2048"
                placeholder="Ctx"
            />
            <button type="submit" disabled={loadingTrainRun || !trainWandbRunEntry?.trim()}>
                {loadingTrainRun ? "..." : "Load"}
            </button>
        </form>

        {#if loadedRun}
            <nav class="tab-navigation">
                <button
                    class="tab-button"
                    class:active={activeTab === "local-attributions"}
                    onclick={() => (activeTab = "local-attributions")}
                >
                    Local Attributions
                </button>
                <button
                    class="tab-button"
                    class:active={activeTab === "activation-contexts"}
                    onclick={() => (activeTab = "activation-contexts")}
                >
                    Activation Contexts
                </button>
            </nav>

            <div
                class="config-wrapper"
                onmouseenter={() => (showConfig = true)}
                onmouseleave={() => (showConfig = false)}
            >
                <button class="config-button">Config</button>
                {#if showConfig}
                    <div class="config-dropdown">
                        <pre>{loadedRun.config_yaml}</pre>
                    </div>
                {/if}
            </div>
        {/if}
    </header>

    <main class="main-content">
        {#if backendError}
            <div class="warning-banner">
                {backendError}
            </div>
        {/if}
        {#if loadedRun && activeTab === "local-attributions"}
            <LocalAttributionsTab />
        {:else if loadedRun && activeTab === "activation-contexts"}
            <ActivationContextsTab />
        {:else if !loadedRun}
            <div class="empty-state">
                <p>Enter a W&B Run ID above to get started</p>
            </div>
        {/if}
    </main>
</div>

<style>
    .app-layout {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    .top-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-bottom: 1px solid #dee2e6;
        flex-shrink: 0;
    }

    .run-input {
        display: flex;
        gap: 0.25rem;
    }

    .run-input input[type="text"] {
        width: 180px;
        padding: 0.4rem 0.6rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .run-input input[type="number"] {
        width: 60px;
        padding: 0.4rem 0.6rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .run-input input[type="text"]:focus,
    .run-input input[type="number"]:focus {
        outline: none;
        border-color: #4a90e2;
    }

    .run-input button {
        padding: 0.4rem 0.75rem;
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 0.85rem;
        cursor: pointer;
        white-space: nowrap;
    }

    .run-input button:hover:not(:disabled) {
        background-color: #357abd;
    }

    .run-input button:disabled {
        background-color: #ccc;
        cursor: not-allowed;
    }

    .tab-navigation {
        display: flex;
        gap: 0.25rem;
    }

    .tab-button {
        padding: 0.4rem 0.75rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        color: #495057;
        transition: all 0.15s ease;
    }

    .tab-button:hover {
        color: #007bff;
        border-color: #007bff;
    }

    .tab-button.active {
        color: white;
        background: #007bff;
        border-color: #007bff;
    }

    .config-wrapper {
        position: relative;
        margin-left: auto;
    }

    .config-button {
        padding: 0.4rem 0.75rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        color: #495057;
    }

    .config-button:hover {
        border-color: #adb5bd;
        background: #f8f9fa;
    }

    .config-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: 0.5rem;
        z-index: 1000;
    }

    .config-dropdown pre {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        max-width: 400px;
        max-height: 70vh;
        overflow: auto;
        margin: 0;
        padding: 0.75rem;
        font-size: 0.75rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .main-content {
        flex: 1;
        min-width: 0;
        min-height: 0;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .warning-banner {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border: 1px solid #ffeeba;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: #6c757d;
    }
</style>
