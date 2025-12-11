<script lang="ts">
    // import { RenderScan } from "svelte-render-scan";
    import type { LoadedRun } from "./lib/api";
    import * as api from "./lib/api";
    import * as attrApi from "./lib/localAttributionsApi";
    import type { ActivationContextsSummary } from "./lib/localAttributionsTypes";
    import { CANONICAL_RUNS, formatRunIdForDisplay } from "./lib/registry";

    import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
    import CorrelationJobStatus from "./components/CorrelationJobStatus.svelte";
    import DatasetSearchTab from "./components/DatasetSearchTab.svelte";
    import LocalAttributionsTab from "./components/LocalAttributionsTab.svelte";
    import ViewSettingsDropdown from "./components/ui/ViewSettingsDropdown.svelte";
    import { onMount } from "svelte";

    let loadingTrainRun = $state(false);

    /** can be a wandb run path, or id. we sanitize this on sumbit */
    let trainWandbRunEntry = $state<string | null>("goodfire/spd/jyo9duz5");
    let contextLength = $state<number | null>(512);

    let loadedRun = $state<LoadedRun | null>(null);
    let backendError = $state<string | null>(null);
    let backendUser = $state<string | null>(null);

    // Lifted activation contexts state - shared between tabs
    let activationContextsSummary = $state<ActivationContextsSummary | null>(null);
    let activationContextsMissing = $state(false);

    // Correlation job state
    let correlationJobStatus = $state<api.CorrelationJobStatus | null>(null);
    let correlationJobSubmitting = $state(false);

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
            loadCorrelationJobStatus();
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
            } else throw e;
        }
    }

    async function loadCorrelationJobStatus() {
        correlationJobStatus = await api.getCorrelationJobStatus();
    }

    async function submitCorrelationJob(params: api.HarvestParams) {
        if (correlationJobSubmitting) return;
        correlationJobSubmitting = true;
        try {
            await api.submitCorrelationJob(params);
            await loadCorrelationJobStatus();
        } finally {
            correlationJobSubmitting = false;
        }
    }

    // Poll correlation job status while pending/running
    $effect(() => {
        const s = correlationJobStatus?.status;
        if (s !== "pending" && s !== "running") return;

        const interval = setInterval(loadCorrelationJobStatus, 1000);
        return () => clearInterval(interval);
    });

    async function loadRun(event: Event) {
        event.preventDefault();
        const input = trainWandbRunEntry?.trim();
        if (!input || !contextLength) return;
        try {
            loadingTrainRun = true;
            loadedRun = null;
            await api.loadRun(input, contextLength);
        } finally {
            // Set loading false before calling loadStatus, otherwise the guard returns early
            loadingTrainRun = false;
        }
        await loadStatus();
    }

    onMount(() => {
        loadStatus();
        api.getWhoami().then((user) => (backendUser = user));
    });

    let activeTab = $state<"prompts" | "activation-contexts" | "dataset-search" | null>(null);
    let showConfig = $state(false);
    let showRegistry = $state(false);

    function selectRegistryEntry(wandbRunId: string) {
        trainWandbRunEntry = wandbRunId;
        showRegistry = false;
    }
</script>

<!-- <RenderScan /> -->
<div class="app-layout">
    <header class="top-bar">
        <span class="backend-user">user: {backendUser ?? "..."}</span>
        <form onsubmit={loadRun} class="run-input">
            <label for="wandb-path">W&B Path/Link:</label>
            <div class="input-with-dropdown">
                <input
                    type="text"
                    id="wandb-path"
                    list="run-options"
                    placeholder="e.g. goodfire/spd/runs/33n6xjjt"
                    bind:value={trainWandbRunEntry}
                    disabled={loadingTrainRun}
                />
                <div
                    class="registry-wrapper"
                    role="group"
                    onmouseenter={() => (showRegistry = true)}
                    onmouseleave={() => (showRegistry = false)}
                >
                    <button type="button" class="registry-button" title="Select from canonical runs"> ▼ </button>
                    {#if showRegistry}
                        <div class="registry-dropdown">
                            {#each CANONICAL_RUNS as entry (entry.wandbRunId)}
                                <button
                                    type="button"
                                    class="registry-entry"
                                    onclick={() => selectRegistryEntry(entry.wandbRunId)}
                                >
                                    <span class="entry-model">{entry.modelName}</span>
                                    <span class="entry-id">{formatRunIdForDisplay(entry.wandbRunId)}</span>
                                    {#if entry.notes}
                                        <span class="entry-notes">{entry.notes}</span>
                                    {/if}
                                </button>
                            {/each}
                        </div>
                    {/if}
                </div>
            </div>
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
        </form>
    </header>

    <nav class="tab-bar">
        <div class="tab-buttons">
            <button
                class="tab-button"
                class:active={activeTab === "dataset-search"}
                onclick={() => (activeTab = "dataset-search")}
            >
                Dataset Search
            </button>
            {#if loadedRun}
                <button
                    class="tab-button"
                    class:active={activeTab === "prompts"}
                    onclick={() => (activeTab = "prompts")}
                >
                    Prompts
                </button>
                <button
                    class="tab-button"
                    class:active={activeTab === "activation-contexts"}
                    onclick={() => (activeTab = "activation-contexts")}
                >
                    Activation Contexts
                </button>
            {/if}
        </div>
        <div class="tab-bar-right">
            <CorrelationJobStatus
                status={correlationJobStatus}
                onSubmit={submitCorrelationJob}
                submitting={correlationJobSubmitting}
            />
            <ViewSettingsDropdown />
            {#if loadedRun}
                <div
                    class="config-wrapper"
                    role="group"
                    onmouseenter={() => (showConfig = true)}
                    onmouseleave={() => (showConfig = false)}
                >
                    <button type="button" class="config-button">Config</button>
                    {#if showConfig}
                        <div class="config-dropdown">
                            <pre>{loadedRun.config_yaml}</pre>
                        </div>
                    {/if}
                </div>
            {/if}
        </div>
    </nav>

    <main class="main-content">
        {#if backendError}
            <div class="warning-banner">
                {backendError}
            </div>
        {/if}
        <!-- Dataset Search tab - always available, doesn't require loaded run -->
        <div class="tab-content" class:hidden={activeTab !== "dataset-search"}>
            <DatasetSearchTab />
        </div>
        {#if loadedRun}
            <!-- Use hidden class instead of conditional rendering to preserve state -->
            <div class="tab-content" class:hidden={activeTab !== "prompts"}>
                <LocalAttributionsTab {activationContextsSummary} {activationContextsMissing} />
            </div>
            <div class="tab-content" class:hidden={activeTab !== "activation-contexts"}>
                <ActivationContextsTab onHarvestComplete={loadActivationContextsSummary} />
            </div>
        {:else if loadingTrainRun}
            <div class="empty-state" class:hidden={activeTab === "dataset-search"}>
                <p>Loading run...</p>
            </div>
        {:else}
            <div class="empty-state" class:hidden={activeTab === "dataset-search"}>
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
        justify-content: space-between;
        align-items: center;
        padding: var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
    }

    .tab-buttons {
        display: flex;
        gap: var(--space-2);
    }

    .tab-bar-right {
        display: flex;
        gap: var(--space-2);
        align-items: center;
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

    .input-with-dropdown {
        display: flex;
        align-items: stretch;
        gap: 0;
    }

    .input-with-dropdown input[type="text"] {
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
        border-right: none;
    }

    .registry-wrapper {
        position: relative;
    }

    .registry-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-top-right-radius: var(--radius-sm);
        border-bottom-right-radius: var(--radius-sm);
        border-top-left-radius: 0;
        border-bottom-left-radius: 0;
        cursor: pointer;
        font-size: var(--text-xs);
        color: var(--text-secondary);
        height: 100%;
        display: flex;
        align-items: center;
    }

    .registry-button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .registry-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: var(--space-1);
        z-index: 1000;
        min-width: 320px;
    }

    .registry-dropdown > :first-child {
        border-top-left-radius: var(--radius-md);
        border-top-right-radius: var(--radius-md);
    }

    .registry-dropdown > :last-child {
        border-bottom-left-radius: var(--radius-md);
        border-bottom-right-radius: var(--radius-md);
    }

    .registry-entry {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-1);
        width: 100%;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-top: none;
        cursor: pointer;
        text-align: left;
        font-family: var(--font-sans);
        border-radius: 0;
    }

    .registry-entry:first-child {
        border-top: 1px solid var(--border-default);
    }

    .registry-entry:hover {
        background: var(--bg-inset);
    }

    .entry-model {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .entry-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
    }

    .entry-notes {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
