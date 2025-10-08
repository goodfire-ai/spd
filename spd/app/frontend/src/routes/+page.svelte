<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type { Status } from "$lib/api";
    import * as api from "$lib/api";
    import { onMount } from "svelte";

    import ActivationContextsTab from "$lib/components/ActivationContextsTab.svelte";
    import NewClusterDashboard from "$lib/components/ClusterDashboardTab.svelte";
    import InterventionsTab from "$lib/components/InterventionsTab.svelte";

    let trainWandbRunId: string | null = null;
    let loadingTrainRun: boolean = false;

    let status: Status | null = null;
    $: availableClusterRuns = status?.train_run?.available_cluster_runs;

    let clusterWandbRunPath: string | null = null; // "goodfire/spd-cluster/wj3xq8ds"; // defaults for dev
    let clusterIteration: number | null = null; // 7000;
    let loadingClusterRun: boolean = false;

    async function loadStatus() {
        console.log("getting status");
        status = await api.getStatus();
        console.log("status:", status);
        if (!status.train_run) {
            return;
        }

        trainWandbRunId = status.train_run.wandb_path.split("/").pop()!;
        availableClusterRuns = status.train_run.available_cluster_runs;

        if (status.cluster_run) {
            clusterWandbRunPath = status.cluster_run.wandb_path;
        }

        if (status.cluster_run?.clustering_shape) {
            const cGroups = status.cluster_run?.clustering_shape.module_component_groups;
            const numComponents = Object.values(cGroups).reduce((acc, val) => acc + val.length, 0);
            console.log(`got ${numComponents} components`);
        }
    }

    async function loadRun() {
        if (!trainWandbRunId?.trim()) return;

        loadingTrainRun = true;
        await api.loadRun(trainWandbRunId);
        loadingTrainRun = false;

        await loadStatus();
    }

    async function loadClusterRun() {
        console.log("loading cluster run", clusterWandbRunPath, clusterIteration);
        const canLoadCluster = clusterWandbRunPath !== null && clusterIteration !== null;
        if (!canLoadCluster) {
            console.log("cannot submit cluster settings", {
                clusterWandbRunPath,
                clusterIteration
            });
            return;
        }

        loadingClusterRun = true;
        await api.loadClusterRun(clusterWandbRunPath!.split("/").pop()!, clusterIteration!);
        loadingClusterRun = false;

        await loadStatus();
    }

    onMount(() => {
        loadStatus();
    });

    let activeTab: "ablation" | "activation-contexts" | "cluster-dashboard" | null = null;
</script>

<div class="app-layout">
    <!-- Left Sidebar -->
    <aside class="sidebar">
        <div class="run-selector">
            <label for="wandb-run-id">W&B Run ID</label>
            <div class="input-group">
                <input
                    type="text"
                    id="wandb-run-id"
                    list="run-options"
                    bind:value={trainWandbRunId}
                    disabled={loadingTrainRun}
                    placeholder="Select or enter run ID"
                />
                <button on:click={loadRun} disabled={loadingTrainRun || !trainWandbRunId?.trim()}>
                    {loadingTrainRun ? "Loading..." : "Load Run"}
                </button>
            </div>
        </div>
        <div class="tab-navigation">
            {#if status?.train_run}
                <button
                    class="tab-button"
                    class:active={activeTab === "activation-contexts"}
                    on:click={() => (activeTab = "activation-contexts")}
                >
                    Activation Contexts
                </button>
                <div class="cluster-settings">
                    <h4>Cluster Settings</h4>
                    <!-- svelte-ignore a11y_consider_explicit_label -->
                    <form on:submit|preventDefault={loadClusterRun}>
                        <label>
                            Clustering Run
                            <select bind:value={clusterWandbRunPath}>
                                {#if availableClusterRuns != null}
                                    {#each availableClusterRuns as run}
                                        <option value={run}>{run}</option>
                                    {/each}
                                {/if}
                            </select>
                        </label>
                        <div class="settings-grid">
                            <label>
                                Iteration
                                <input type="number" bind:value={clusterIteration} />
                            </label>
                        </div>
                        <button class="cluster-load" type="submit" disabled={loadingClusterRun}
                            >Load Cluster Run</button
                        >
                    </form>
                </div>
                <button
                    class="tab-button"
                    class:active={activeTab === "ablation"}
                    disabled={status?.cluster_run == null}
                    on:click={() => (activeTab = "ablation")}
                >
                    Component Ablation
                    <div class="spinner" class:hidden={!loadingClusterRun}></div>
                </button>
                <button
                    class="tab-button"
                    class:active={activeTab === "cluster-dashboard"}
                    disabled={status?.cluster_run == null}
                    on:click={() => (activeTab = "cluster-dashboard")}
                >
                    Cluster Dashboard
                    <div class="spinner" class:hidden={!loadingClusterRun}></div>
                </button>
            {/if}
        </div>
    </aside>

    <!-- Main Content -->
    <div class="main-content">
        {#if status?.train_run}
            <div class:hidden={activeTab !== "activation-contexts"}>
                <ActivationContextsTab
                    availableComponentLayers={status.train_run.component_layers}
                />
            </div>
            <div class:hidden={activeTab !== "ablation"}>
                {#if status?.cluster_run && clusterIteration !== null}
                    <InterventionsTab
                        cluster_run={status.cluster_run}
                        iteration={clusterIteration}
                    />
                {:else}
                    <div class="status">No cluster run selected.</div>
                {/if}
            </div>
            <div class:hidden={activeTab !== "cluster-dashboard"}>
                {#if status?.cluster_run && clusterIteration !== null}
                    <NewClusterDashboard iteration={clusterIteration} />
                {:else}
                    <div class="status">No cluster run selected.</div>
                {/if}
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
</style>
