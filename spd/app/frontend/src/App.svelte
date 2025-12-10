<script lang="ts">
    // import { RenderScan } from "svelte-render-scan";
    import type { LoadedRun } from "./lib/api";
    import * as api from "./lib/api";
    import * as attrApi from "./lib/localAttributionsApi";
    import type { ActivationContextsSummary } from "./lib/localAttributionsTypes";

    import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
    import CorrelationJobStatus from "./components/CorrelationJobStatus.svelte";
    import LocalAttributionsTab from "./components/LocalAttributionsTab.svelte";
    import { onMount } from "svelte";

    let loadingTrainRun = $state(false);

    /** can be a wandb run path, or id. we sanitize this on sumbit */
    let trainWandbRunEntry = $state<string | null>("goodfire/spd/vjbol27n");
    let contextLength = $state<number | null>(512);

    let loadedRun = $state<LoadedRun | null>(null);
    let backendError = $state<string | null>(null);
    let backendUser = $state<string | null>(null);

    // Lifted activation contexts state - shared between tabs
    let activationContextsSummary = $state<ActivationContextsSummary | null>(null);
    let activationContextsMissing = $state(false);

    // Reference to correlation job component for reloading
    let correlationJobStatus: CorrelationJobStatus;

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
            contextLength = loadedRun.context_length;

            // Load activation contexts summary and correlation status
            loadActivationContextsSummary();
            correlationJobStatus?.reload();
        } catch (error) {
            console.error("error loading status", error);
            // if the backend is down, we keep the local state
            backendError = "Backend unreachable. Showing cached view.";
        }
    }

    async function loadActivationContextsSummary() {
        try {
            activationContextsSummary = await attrApi.getActivationContextsSummary();
            activationContextsMissing = false;
        } catch (e) {
            const status = (e as { status?: number }).status;
            if (status === 404) {
                activationContextsMissing = true;
                activationContextsSummary = null;
            }
        }
    }

    async function loadRun(event: Event) {
        event.preventDefault();
        const input = trainWandbRunEntry?.trim();
        if (!input || !contextLength) return;
        try {
            loadingTrainRun = true;
            loadedRun = null;
            await api.loadRun(input, contextLength);
            // Set loading false before calling loadStatus, otherwise the guard returns early
            loadingTrainRun = false;
            await loadStatus();
        } catch (error) {
            console.error("error loading run", error);
            loadingTrainRun = false;
        }
    }

    onMount(() => {
        loadStatus();
        api.getWhoami().then((user) => (backendUser = user));
    });

    let activeTab = $state<"prompts" | "activation-contexts" | null>(null);
    let showConfig = $state(false);
</script>

<!-- <RenderScan /> -->
<div class="app-layout">
    <header class="top-bar">
        <span class="backend-user">user: {backendUser ?? "..."}</span>
        <form onsubmit={loadRun} class="run-input">
            <label for="wandb-path">W&B Path/Link:</label>
            <input
                type="text"
                id="wandb-path"
                list="run-options"
                placeholder="e.g. goodfire/spd/runs/33n6xjjt"
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
            <button class="load-button" type="submit" disabled={loadingTrainRun || !trainWandbRunEntry?.trim()}>
                {loadingTrainRun ? "..." : "Load"}
            </button>
            {#if loadedRun}
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
        </form>
    </header>

    {#if loadedRun}
        <nav class="tab-bar">
            <button class="tab-button" class:active={activeTab === "prompts"} onclick={() => (activeTab = "prompts")}>
                Prompts
            </button>
            <button
                class="tab-button"
                class:active={activeTab === "activation-contexts"}
                onclick={() => (activeTab = "activation-contexts")}
            >
                Activation Contexts
            </button>
            <CorrelationJobStatus bind:this={correlationJobStatus} />
        </nav>
    {/if}

    <main class="main-content">
        {#if backendError}
            <div class="warning-banner">
                {backendError}
            </div>
        {/if}
        {#if loadedRun}
            <!-- Use hidden class instead of conditional rendering to preserve state -->
            <div class="tab-content" class:hidden={activeTab !== "prompts"}>
                <LocalAttributionsTab {activationContextsSummary} {activationContextsMissing} />
            </div>
            <div class="tab-content" class:hidden={activeTab !== "activation-contexts"}>
                <ActivationContextsTab onHarvestComplete={loadActivationContextsSummary} />
            </div>
        {:else if loadingTrainRun}
            <div class="empty-state">
                <p>Loading run...</p>
            </div>
        {:else}
            <div class="empty-state">
                <p>Enter a W&B Path above to get started</p>
            </div>
        {/if}
    </main>
</div>

<style>
    .app-layout {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        background: var(--bg-base);
    }

    .top-bar {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
    }

    .run-input {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .run-input label {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        white-space: nowrap;
        font-family: var(--font-sans);
        font-weight: 500;
    }

    .run-input input[type="text"] {
        width: 250px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .run-input input[type="text"]::placeholder {
        color: var(--text-muted);
    }

    .run-input input[type="number"] {
        width: 70px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .run-input input[type="text"]:focus,
    .run-input input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .load-button {
        padding: var(--space-1) var(--space-3);
        color: var(--text-primary);
        border: 1px solid var(--border-default);
        font-weight: 500;
        white-space: nowrap;
    }

    .load-button:hover:not(:disabled) {
        background: var(--accent-primary);
        color: white;
    }

    .load-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
    }

    .tab-button {
        padding: var(--space-1) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        font-weight: 500;
        color: var(--text-secondary);
    }

    .tab-button:hover {
        color: var(--text-primary);
        border-color: var(--border-strong);
        background: var(--bg-inset);
    }

    .tab-button.active {
        color: white;
        background: var(--accent-primary);
        border-color: var(--accent-primary);
    }

    .config-wrapper {
        position: relative;
    }

    .config-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .config-button:hover {
        background: var(--bg-inset);
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
        border-radius: var(--radius-md);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
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

    .backend-user {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .tab-bar {
        display: flex;
        gap: var(--space-2);
        padding: var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
    }

    .main-content {
        flex: 1;
        min-width: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-content {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-content.hidden {
        display: none;
    }

    .warning-banner {
        background: var(--bg-surface);
        color: var(--accent-primary);
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--accent-primary-dim);
        border-radius: var(--radius-md);
        margin: var(--space-3);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
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
