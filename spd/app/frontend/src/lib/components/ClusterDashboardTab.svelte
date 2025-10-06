<script lang="ts">
    import { onDestroy, onMount } from "svelte";
    import { getClusterDashboardData, type ClusterDashboardResponse } from "$lib/api";
    import ClusterDashboardBody from "./ClusterDashboardBody.svelte";

    export let iteration: number;

    let loading = true;
    let errorMsg: string | null = null;
    let dashboard: ClusterDashboardResponse | null = null;

    let nSamples = 16;
    let nBatches = 2;
    let batchSize = 64;
    let contextLength = 64;

    let pendingController: AbortController | null = null;

    async function fetchDashboard() {
        pendingController?.abort();
        const controller = new AbortController();
        pendingController = controller;

        loading = true;
        errorMsg = null;

        try {
            console.log("fetching dashboard");
            dashboard = await getClusterDashboardData({
                iteration,
                n_samples: nSamples,
                n_batches: nBatches,
                batch_size: batchSize,
                context_length: contextLength,
                // signal: controller.signal
            });
        } catch (e: any) {
            if (controller.signal.aborted) return;
            errorMsg = e?.message ?? String(e);
        } finally {
            if (!controller.signal.aborted) {
                loading = false;
            }
        }
    }

    onMount(() => {
        fetchDashboard();
    });

    onDestroy(() => {
        pendingController?.abort();
        pendingController = null;
    });

    function refresh() {
        fetchDashboard().catch((e) => (errorMsg = e?.message ?? String(e)));
    }
</script>

<div class="tab-content">
    <div class="toolbar">
        <form class="toolbar-form" on:submit|preventDefault={refresh}>
            <!-- <label>
                Iteration
                <input type="number" bind:value={iteration} />
            </label> -->
            <label>
                Samples
                <input type="number" min={1} bind:value={nSamples} />
            </label>
            <label>
                Batches
                <input type="number" min={1} bind:value={nBatches} />
            </label>
            <label>
                Batch Size
                <input type="number" min={1} bind:value={batchSize} />
            </label>
            <label>
                Context
                <input type="number" min={1} bind:value={contextLength} />
            </label>
            <button class="run-button" type="submit">Run</button>
        </form>
    </div>

    {#if loading}
        <div class="status">Loading...</div>
    {:else if errorMsg}
        <div class="status-error">{errorMsg}</div>
    {:else if dashboard}
        <ClusterDashboardBody {dashboard} />
    {/if}
</div>

<style>
    .tab-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }

    .toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: flex-end;
    }

    .toolbar-form {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: flex-end;
    }

    .toolbar-form label {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 0.85rem;
        color: #555;
    }

    .toolbar-form input {
        width: 90px;
        padding: 4px 6px;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .run-button {
        padding: 6px 12px;
        border-radius: 4px;
        border: 1px solid #0d6efd;
        background: #0d6efd;
        color: #fff;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .run-button:hover {
        background: #0b5ed7;
        border-color: #0b5ed7;
    }

    .status {
        color: #333;
    }

    .status-error {
        color: #b00020;
    }
</style>
