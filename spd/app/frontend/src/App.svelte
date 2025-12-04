<script lang="ts">
    import { RenderScan } from "svelte-render-scan";
    import type { LoadedRun } from "./lib/api";
    import * as api from "./lib/api";

    import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
    import LocalAttributionsTab from "./components/LocalAttributionsTab.svelte";
    import { parseWandbRunPath } from "./lib";
    import { onMount } from "svelte";

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

    onMount(loadStatus);

    let activeTab = $state<"activation-contexts" | "local-attributions" | null>(null);
    let showConfig = $state(false);
</script>

<RenderScan />
<div class="app-layout">
    <header class="top-bar">
        <form onsubmit={loadRun} class="run-input">
            <label for="wandb-run-id">W&B Run ID:</label>
            <input
                type="text"
                id="wandb-run-id"
                list="run-options"
                bind:value={trainWandbRunEntry}
                disabled={loadingTrainRun}
            />
            <label for="context-length">Context Length:</label>
            <input
                type="number"
                id="context-length"
                bind:value={contextLength}
                disabled={loadingTrainRun}
                min="1"
                max="2048"
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

            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div
                class="config-wrapper"
                role="group"
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
        background: var(--bg-base);
    }

    .top-bar {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
    }

    .run-input {
        display: flex;
        gap: var(--space-1);
    }

    /* CHECK */
    .run-input label {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        white-space: nowrap;
        font-family: var(--font-mono);
    }

    .run-input input[type="text"] {
        width: 200px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .run-input input[type="text"]::placeholder {
        color: var(--text-muted);
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
        border-color: var(--accent-warm-dim);
    }

    .run-input button {
        padding: var(--space-1) var(--space-3);
        background: var(--accent-warm);
        color: var(--bg-base);
        border: none;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .run-input button:hover:not(:disabled) {
        background: var(--text-primary);
    }

    .run-input button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .tab-navigation {
        display: flex;
        gap: var(--space-1);
    }

    .tab-button {
        padding: var(--space-1) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        cursor: pointer;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .tab-button:hover {
        color: var(--text-primary);
        border-color: var(--border-strong);
        background: var(--bg-inset);
    }

    .tab-button.active {
        color: var(--bg-base);
        background: var(--accent-warm);
        border-color: var(--accent-warm);
    }

    .config-wrapper {
        position: relative;
        margin-left: auto;
    }

    .config-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        cursor: pointer;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .config-button:hover {
        border-color: var(--border-strong);
        color: var(--text-primary);
    }

    .config-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: var(--space-2);
        z-index: 1000;
    }

    .config-dropdown pre {
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        max-width: 420px;
        max-height: 70vh;
        overflow: auto;
        margin: 0;
        padding: var(--space-3);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-primary);
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .main-content {
        flex: 1;
        min-width: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .warning-banner {
        background: var(--bg-elevated);
        color: var(--accent-warm);
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--accent-warm-dim);
        border-left: 3px solid var(--accent-warm);
        margin: var(--space-3);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        flex-shrink: 0;
    }

    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: var(--text-muted);
        font-family: var(--font-sans);
    }
</style>
