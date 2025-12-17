<script lang="ts">
    import type { RunState } from "./lib/api";
    import * as api from "./lib/api";
    import * as attrApi from "./lib/localAttributionsApi";
    import type { ActivationContextsSummary } from "./lib/localAttributionsTypes";

    import { onMount } from "svelte";
    import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
    import ClusterPathInput from "./components/ClusterPathInput.svelte";
    import CorrelationJobStatus from "./components/CorrelationJobStatus.svelte";
    import DatasetSearchTab from "./components/DatasetSearchTab.svelte";
    import LocalAttributionsTab from "./components/LocalAttributionsTab.svelte";
    import RunSelector from "./components/RunSelector.svelte";
    import DisplaySettingsDropdown from "./components/ui/DisplaySettingsDropdown.svelte";
    import type { Loadable } from "./lib";

    let runState = $state<Loadable<RunState | null>>(null);
    let backendUser = $state<Loadable<string>>(null);

    // When true, show the run selector even if a run is loaded
    let showRunSelector = $state(true);

    // Lifted activation contexts state - shared between tabs
    let activationContextsSummary = $state<ActivationContextsSummary | null>(null);

    async function loadStatus() {
        try {
            const newLoadedRun = await api.getStatus();

            // if we had a run loaded, but the backend says null, then we're out of sync (backend likely restarted)
            if (runState?.status === "loaded" && runState.data !== null && newLoadedRun === null) {
                runState = { status: "error", error: "Backend state lost (restarted). Showing cached view." };
                return;
            }

            runState = { status: "loaded", data: newLoadedRun };

            if (newLoadedRun) {
                showRunSelector = false;
                loadActivationContextsSummary();
            }
        } catch (error) {
            console.error("error loading status", error);
            runState = { status: "error", error: "Backend unreachable. Showing cached view." };
        }
    }

    async function loadRun(wandbPath: string, contextLength: number) {
        runState = { status: "loading" };
        await api.loadRun(wandbPath, contextLength);
        await loadStatus();
    }

    async function loadActivationContextsSummary() {
        activationContextsSummary = await attrApi.getActivationContextsSummary();
    }

    function handleChangeRun() {
        showRunSelector = true;
    }

    onMount(() => {
        loadStatus();
        api.getWhoami().then((user) => (backendUser = { status: "loaded", data: user }));
    });

    let activeTab = $state<"prompts" | "components" | "dataset-search" | null>(null);
    let showRunMenu = $state(false);
</script>

{#if showRunSelector}
    <RunSelector
        onSelect={loadRun}
        isLoading={runState?.status === "loading"}
        username={backendUser?.status === "loaded" ? backendUser.data : null}
    />
{:else}
    <div class="app-layout">
        <header class="top-bar">
            {#if runState?.status === "loaded" && runState.data}
                <div
                    class="run-menu"
                    onmouseenter={() => (showRunMenu = true)}
                    onmouseleave={() => (showRunMenu = false)}
                >
                    <button type="button" class="run-menu-trigger">
                        <span class="run-path">{runState.data.wandb_path}</span>
                    </button>
                    {#if showRunMenu}
                        <div class="run-menu-dropdown">
                            <pre class="config-yaml">{runState.data.config_yaml}</pre>
                            <button type="button" class="change-run-button" onclick={handleChangeRun}>Change Run</button
                            >
                        </div>
                    {/if}
                </div>
            {/if}

            <nav class="nav-group">
                <button
                    type="button"
                    class="tab-button"
                    class:active={activeTab === "dataset-search"}
                    onclick={() => (activeTab = "dataset-search")}
                >
                    Dataset Search
                </button>
                {#if runState?.status === "loaded" && runState.data}
                    <button
                        type="button"
                        class="tab-button"
                        class:active={activeTab === "prompts"}
                        onclick={() => (activeTab = "prompts")}
                    >
                        Prompts
                    </button>
                    <button
                        type="button"
                        class="tab-button"
                        class:active={activeTab === "components"}
                        onclick={() => (activeTab = "components")}
                    >
                        Components
                    </button>
                    <ClusterPathInput runState={runState.data} />
                {/if}
            </nav>

            <div class="top-bar-spacer"></div>
            <DisplaySettingsDropdown />
        </header>

        <main class="main-content">
            {#if runState?.status === "error"}
                <div class="warning-banner">
                    {runState.error}
                </div>
            {/if}
            <!-- Dataset Search tab - always available, doesn't require loaded run -->
            <div class="tab-content" class:hidden={activeTab !== "dataset-search"}>
                <DatasetSearchTab />
            </div>
            {#if runState?.status === "loaded"}
                <!-- Use hidden class instead of conditional rendering to preserve state -->
                <div class="tab-content" class:hidden={activeTab !== "prompts"}>
                    <LocalAttributionsTab {activationContextsSummary} />
                </div>
                <div class="tab-content" class:hidden={activeTab !== "components"}>
                    <ActivationContextsTab {activationContextsSummary} />
                </div>
            {:else if runState?.status === "loading"}
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
{/if}

<style>
    .app-layout {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        background: var(--bg-base);
    }

    .top-bar {
        display: flex;
        align-items: stretch;
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
        min-height: 44px;
    }

    /* Run menu - hoverable dropdown */
    .run-menu {
        position: relative;
        display: flex;
        align-items: stretch;
    }

    .run-menu-trigger {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: 0 var(--space-3);
        margin: 0;
        background: none;
        border: none;
        border-right: 1px solid var(--border-default);
        border-radius: 0;
        cursor: pointer;
        font: inherit;
        font-size: var(--text-sm);
        transition: background 0.15s;
    }

    .run-menu-trigger:hover .run-path {
        background: var(--bg-inset);
    }

    .run-path {
        font-family: var(--font-mono);
        color: var(--text-primary);
    }

    .run-menu-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding: var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    .config-yaml {
        max-width: 420px;
        max-height: 50vh;
        overflow: auto;
        margin: 0;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-primary);
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .change-run-button {
        padding: var(--space-2) var(--space-3);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
        cursor: pointer;
        text-align: center;
    }

    .change-run-button:hover {
        background: var(--bg-surface);
        color: var(--text-primary);
    }

    /* Navigation tabs */
    .nav-group {
        display: flex;
    }

    .tab-button {
        padding: var(--space-2) var(--space-4);
        margin: 0;
        background: none;
        border: none;
        border-right: 1px solid var(--border-default);
        border-radius: 0;
        font: inherit;
        font-weight: 500;
        font-size: var(--text-sm);
        color: var(--text-muted);
        cursor: pointer;
        transition:
            color 0.15s,
            background 0.15s;
    }

    .tab-button:hover {
        color: var(--text-primary);
        background: var(--bg-inset);
    }

    .tab-button.active {
        color: var(--text-primary);
        background: var(--bg-inset);
    }

    .top-bar-spacer {
        flex: 1;
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
