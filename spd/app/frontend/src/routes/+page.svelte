<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import type { Status } from "$lib/api";
    import { onMount } from "svelte";
    import * as api from "$lib/api";

    import ActivationContextsTab from "$lib/components/ActivationContextsTab.svelte";
    import Interventions from "$lib/components/Interventions.svelte";
    import RunSelector from "$lib/components/RunSelector.svelte";
    import ClusterDashboard from "$lib/components/ClusterDashboard.svelte";
    import NewClusterDashboard from "$lib/components/NewClusterDashboard.svelte";

    let isLoading = false;
    let wandbRunId: string | null = null;
    let loadingRun = false;
    let activeTab: "ablation" | "activation-contexts" | "cluster-dashboard" | "cluster-dashboard-new" = "ablation";

    let status: Status | null = null;
    async function getStatus() {
        console.log("getting status");
        status = await api.getStatus();
        wandbRunId = status.run_id
        console.log("status", status);
    }

    onMount(() => {
        getStatus();
    });
</script>

<main>
    <div class="app-layout">
        <!-- Left Sidebar -->
        <aside class="sidebar">
            <RunSelector bind:loadingRun bind:wandbRunId {isLoading} />

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
                    class:active={activeTab === "cluster-dashboard"}
                    on:click={() => (activeTab = "cluster-dashboard")}
                >
                    Cluster Dashboard
                </button>
                <button
                    class="tab-button"
                    class:active={activeTab === "cluster-dashboard-new"}
                    on:click={() => (activeTab = "cluster-dashboard-new")}
                >
                    Cluster Dashboard (new)
                </button>
            </div>
        </aside>

        <!-- Main Content -->
        <div class="main-content">
            {#if status}
                <div
                    class="activation-contexts-container"
                    class:hidden={activeTab !== "activation-contexts"}
                >
                    <ActivationContextsTab availableComponentLayers={status.component_layers} />
                </div>
            {/if}

            <div class:hidden={activeTab !== "ablation"}>
                <Interventions />
            </div>

            <div class:hidden={activeTab !== "cluster-dashboard"}>
                <div class="activation-contexts-container">
                    <ClusterDashboard runId={status?.run_id ?? null} />
                </div>
            </div>

            <div class:hidden={activeTab !== "cluster-dashboard-new"}>
                <div class="activation-contexts-container">
                    <NewClusterDashboard runId={status?.run_id ?? null} />
                </div>
            </div>
        </div>
    </div>
</main>

<style>
    main {
        font-family:
            system-ui,
            -apple-system,
            sans-serif;
    }

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

    .activation-contexts-container {
        padding: 1rem;
        background: white;
        border-radius: 8px;
        min-height: 60vh;
    }
</style>
