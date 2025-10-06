<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type { Status } from "$lib/api";
    import { onMount } from "svelte";
    import * as api from "$lib/api";

    import ActivationContextsTab from "$lib/components/ActivationContextsTab.svelte";
    import InterventionsTab from "$lib/components/InterventionsTab.svelte";
    import RunSelector from "$lib/components/RunSelector.svelte";
    import NewClusterDashboard from "$lib/components/ClusterDashboardTab.svelte";

    let isLoading = false;
    let trainWandbRunPath: string | null = null;

    let loadingRun = false;
    let availableClusterRuns: string[] | null = null;

    let clusterWandbRunPath: string | null = null;

    let status: Status | null = null;

    async function getStatus() {
        console.log("getting status");
        status = await api.getStatus();
        console.log("status:", status);

        trainWandbRunPath = status?.train_run?.wandb_path ?? null;
        clusterWandbRunPath = status?.cluster_run?.wandb_path ?? null;
        availableClusterRuns = status?.train_run?.available_cluster_runs ?? [];
    }

    onMount(() => {
        getStatus();
    });

    let clusterIteration: number | null = null;
    let savingClusterSettings = false;
    $: canSubmitClusterSettings = clusterWandbRunPath !== null && clusterIteration !== null;

    async function loadClusterRun() {
        console.log("loading cluster run", clusterWandbRunPath, clusterIteration);
        if (!canSubmitClusterSettings) {
            console.log("cannot submit cluster settings", clusterWandbRunPath, clusterIteration);
            return;
        }

        savingClusterSettings = true;
        try {
            await api.loadClusterRun(clusterWandbRunPath!.split("/").pop()!, clusterIteration!);
            // clusterWandbRunPath = null; // clusterIteration = null;
            await getStatus();
        } catch (error) {
            console.error(error);
        } finally {
            savingClusterSettings = false;
        }
    }

    let activeTab: "ablation" | "activation-contexts" | "cluster-dashboard-new" = "ablation";
</script>

<div class="app-layout">
    <!-- Left Sidebar -->
    <aside class="sidebar">
        <RunSelector bind:loadingRun bind:trainWandbRunPath {isLoading} />

        {#if status?.train_run}
            <div class="cluster-settings">
                <h4>Cluster Settings</h4>
                <form on:submit|preventDefault={loadClusterRun}>
                    <label>
                        Clustering Run
                        <select bind:value={clusterWandbRunPath}>
                            {#if availableClusterRuns !== null}
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
                        <!-- <label>
                            Samples
                            <input type="number" min={1} bind:value={clusterNSamples} />
                        </label>
                        <label>
                            Batches
                            <input type="number" min={1} bind:value={clusterNBatches} />
                        </label>
                        <label>
                            Batch Size
                            <input type="number" min={1} bind:value={clusterBatchSize} />
                        </label>
                        <label>
                            Context
                            <input type="number" min={1} bind:value={clusterContextLength} />
                        </label> -->
                    </div>
                    <button class="cluster-save" type="submit" disabled={savingClusterSettings}>
                        {savingClusterSettings ? "Saving..." : "Load Cluster"}
                    </button>
                </form>
            </div>
        {/if}

        <div class="tab-navigation">
            <button
                class="tab-button"
                class:active={activeTab === "ablation"}
                on:click={() => (activeTab = "ablation")}
            >
                Component Ablation
            </button>
            <button
                class="tab-button"
                class:active={activeTab === "activation-contexts"}
                on:click={() => (activeTab = "activation-contexts")}
            >
                Activation Contexts
            </button>
            <button
                class="tab-button"
                class:active={activeTab === "cluster-dashboard-new"}
                on:click={() => (activeTab = "cluster-dashboard-new")}
            >
                Cluster Dashboard
            </button>
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
                <InterventionsTab />
            </div>

            {#if status?.cluster_run}
                <div class:hidden={activeTab !== "cluster-dashboard-new"}>
                    <NewClusterDashboard iteration={clusterIteration!} />
                </div>
            {/if}
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
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
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
        gap: 1rem;
    }

    .tab-navigation {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .tab-button {
        padding: 0.75rem 1rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 500;
        color: #6c757d;
        transition: all 0.2s;
        text-align: left;
    }

    .tab-button:hover {
        color: #007bff;
        background: #f0f8ff;
        border-color: #007bff;
    }

    .tab-button.active {
        color: white;
        background: #007bff;
        border-color: #0056b3;
    }

    .cluster-settings {
        padding: 1rem;
        background: white;
        border-radius: 8px;
    }

    .cluster-save {
        padding: 0.5rem 1rem;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
    }

    .cluster-save:hover {
        background: #0056b3;
        cursor: pointer;
    }
</style>
