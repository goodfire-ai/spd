<script lang="ts">
    import { onDestroy, onMount } from "svelte";
    import {
        getClusterDashboardData,
        type ClusterDashboardResponse,
        type ClusterDataDTO
    } from "$lib/api";
    import ClusterDetailNew from "$lib/components/ClusterDetailNew.svelte";
    import MiniModelView from "$lib/components/MiniModelView.svelte";
    import Sparkbars from "$lib/components/Sparkbars.svelte";
    
    type ClusterRow = {
        id: number;
        clusterHash: string;
        componentCount: number;
        modules: string[];
    };

    type ClusterMap = Record<string, ClusterDataDTO>;

    export let runId: string | null = null;

    let loading = true;
    let errorMsg: string | null = null;
    let rows: ClusterRow[] = [];
    let clusterMap: ClusterMap = {};
    let dashboard: ClusterDashboardResponse | null = null;

    let iteration = 3000;
    let nSamples = 16;
    let nBatches = 2;
    let batchSize = 64;
    let contextLength = 64;

    let showDetail = false;
    let currentCluster: ClusterDataDTO | null = null;
    let sortKey: "id" | "componentCount" = "id";
    let sortDir: "asc" | "desc" = "asc";

    let pendingController: AbortController | null = null;

    function buildRows(clusters: ClusterDataDTO[]): ClusterRow[] {
        return clusters.map((cluster, idx) => {
            const parts = cluster.cluster_hash.split("-");
            const maybeId = Number.parseInt(parts[parts.length - 1] ?? "", 10);
            const id = Number.isNaN(maybeId) ? idx : maybeId;
            const modules = new Set<string>();
            cluster.components?.forEach((c) => modules.add(c.module));
            return {
                id,
                clusterHash: cluster.cluster_hash,
                componentCount: cluster.components?.length ?? 0,
                modules: Array.from(modules)
            } satisfies ClusterRow;
        });
    }

    function applySort(nextRows: ClusterRow[]): ClusterRow[] {
        const sorted = [...nextRows];
        if (sortKey === "id") {
            sorted.sort((a, b) => (sortDir === "asc" ? a.id - b.id : b.id - a.id));
        } else {
            sorted.sort((a, b) =>
                sortDir === "asc" ? a.componentCount - b.componentCount : b.componentCount - a.componentCount
            );
        }
        return sorted;
    }

    function resetDetail() {
        showDetail = false;
        currentCluster = null;
    }

    async function fetchDashboard() {
        pendingController?.abort();
        const controller = new AbortController();
        pendingController = controller;

        loading = true;
        errorMsg = null;

        try {
            const result = await getClusterDashboardData({
                iteration,
                n_samples: nSamples,
                n_batches: nBatches,
                batch_size: batchSize,
                context_length: contextLength,
                signal: controller.signal
            });
            if (controller.signal.aborted) return;

            dashboard = result;
            clusterMap = Object.fromEntries(result.clusters.map((cluster) => [cluster.cluster_hash, cluster]));
            rows = applySort(buildRows(result.clusters));
            resetDetail();
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
        fetchDashboard().catch((e) => (errorMsg = e?.message ?? String(e)));
    });

    onDestroy(() => {
        pendingController?.abort();
        pendingController = null;
    });

    function toggleSort(key: "id" | "componentCount") {
        if (sortKey === key) {
            sortDir = sortDir === "asc" ? "desc" : "asc";
        } else {
            sortKey = key;
            sortDir = "asc";
        }
        rows = applySort(rows);
    }

    function onView(row: ClusterRow) {
        const cluster = clusterMap[row.clusterHash];
        if (!cluster) return;
        currentCluster = cluster;
        showDetail = true;
    }

    function refresh() {
        fetchDashboard().catch((e) => (errorMsg = e?.message ?? String(e)));
    }

    const formatValue = (value: unknown): string => {
        if (value === null || value === undefined) return "";
        if (typeof value === "number") return value.toLocaleString();
        if (typeof value === "string") return value;
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    };

    $: modelInfo = dashboard?.model_info ?? null;
    $: modelStats = dashboard
        ? (
              [
                  { label: "Cluster Run", value: dashboard.cluster_run_path },
                  { label: "Cluster Run ID", value: dashboard.run_id },
                  { label: "Iteration", value: dashboard.iteration },
                  { label: "Model Run", value: modelInfo?.model_path },
                  { label: "Tokenizer", value: modelInfo?.tokenizer_name },
                  { label: "Total Modules", value: modelInfo?.total_modules },
                  { label: "Total Components", value: modelInfo?.total_components },
                  { label: "Total Clusters", value: modelInfo?.total_clusters },
                  { label: "Parameters", value: modelInfo?.total_parameters },
                  { label: "Trainable Parameters", value: modelInfo?.trainable_parameters }
              ] as const
          ).filter(({ value }) => value !== undefined && value !== null && value !== "")
        : runId
          ? [{ label: "Run", value: runId }]
          : [];
</script>

<div class="tab-content">
    <div class="toolbar">
        <form class="toolbar-form" on:submit|preventDefault={refresh}>
            <label>
                Iteration
                <input type="number" bind:value={iteration} />
            </label>
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
            <button class="refresh-button" type="submit">Run</button>
        </form>
        <button class="refresh-button secondary" on:click={refresh}>Refresh</button>
    </div>

    {#if loading}
        <div class="status">Loading...</div>
    {:else if errorMsg}
        <div class="status-error">{errorMsg}</div>
    {:else if !showDetail}
        <div class="table-wrapper">
            {#if modelStats.length}
                <div class="model-info">
                    {#each modelStats as stat}
                        <div>
                            <span class="model-info-label">{stat.label}:</span>
                            <span class="model-info-value">{formatValue(stat.value)}</span>
                        </div>
                    {/each}
                </div>
            {/if}
            <table class="cluster-table">
                <thead>
                    <tr>
                        <th class="col-id">
                            <button
                                class="th-btn"
                                on:click={() => toggleSort("id")}
                            >ID</button
                            >
                        </th>
                        <th class="col-comps">
                            <button
                                class="th-btn"
                                on:click={() => toggleSort("componentCount")}
                            >Comps</button
                            >
                        </th>
                        <th class="col-model">Model View</th>
                        <th class="col-hist">All Activations</th>
                        <th class="col-hist">Max Activation Pos</th>
                        <th class="col-modules">Modules</th>
                        <th class="col-actions">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {#each rows as row}
                        <tr>
                            <td class="col-id">{row.id}</td>
                            <td class="col-comps">{row.componentCount}</td>
                            <td class="col-model">
                                <MiniModelView
                                    components={clusterMap[row.clusterHash]?.components ?? []}
                                />
                            </td>
                            <td class="col-hist">
                                {#if clusterMap[row.clusterHash]?.stats?.all_activations?.bin_counts}
                                    <Sparkbars
                                        bins={clusterMap[row.clusterHash]!.stats!.all_activations!
                                            .bin_counts as number[]}
                                        width={160}
                                        height={48}
                                        color="#4169E1"
                                    />
                                {:else}
                                    <div class="hist-placeholder"></div>
                                {/if}
                            </td>
                            <td class="col-hist">
                                {#if clusterMap[row.clusterHash]?.stats?.max_activation_position?.bin_counts}
                                    <Sparkbars
                                        bins={clusterMap[row.clusterHash]!.stats!
                                            .max_activation_position!.bin_counts as number[]}
                                        width={160}
                                        height={48}
                                        color="#DC143C"
                                    />
                                {:else}
                                    <div class="hist-placeholder"></div>
                                {/if}
                            </td>
                            <td class="col-modules">{row.modules.join(", ")}</td>
                            <td class="col-actions">
                                <button class="view-button" on:click={() => onView(row)}
                                    >View →</button
                                >
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {:else}
    <div class="detail-toolbar">
        <button class="back-button" on:click={resetDetail}>← Back</button>
        </div>
        <ClusterDetailNew cluster={currentCluster} />
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

    .refresh-button {
        padding: 6px 12px;
        border-radius: 4px;
        border: 1px solid #0d6efd;
        background: #0d6efd;
        color: #fff;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .refresh-button:hover {
        background: #0b5ed7;
        border-color: #0b5ed7;
    }

    .refresh-button.secondary {
        border-color: #dee2e6;
        background: #fff;
        color: #0d6efd;
    }

    .refresh-button.secondary:hover {
        background: #e9ecef;
    }

    .status {
        color: #333;
    }

    .status-error {
        color: #b00020;
    }

    .table-wrapper {
        overflow: auto;
    }

    .model-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        padding: 8px 0 12px 0;
        color: #555;
    }

    .model-info-label {
        font-weight: 600;
        margin-right: 0.35rem;
        color: #333;
    }

    .model-info-value {
        color: #343a40;
    }

    .cluster-table {
        width: 100%;
        border-collapse: collapse;
    }

    .cluster-table th,
    .cluster-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }

    .cluster-table th {
        background: #f8f9fa;
        font-weight: 600;
        text-align: left;
    }

    .cluster-table th.col-comps,
    .cluster-table td.col-comps {
        text-align: right;
    }

    .col-hist {
        width: 170px;
    }
    .hist-placeholder {
        width: 160px;
        height: 48px;
        background: #f1f3f5;
        border: 1px dashed #dee2e6;
        border-radius: 4px;
    }

    .view-button {
        padding: 4px 8px;
        border: 1px solid #dee2e6;
        background: #fff;
        border-radius: 4px;
        cursor: pointer;
    }

    .view-button:hover {
        background: #f8f9fa;
    }

    .th-btn {
        font: inherit;
        background: transparent;
        border: none;
        cursor: pointer;
        padding: 0;
        color: #333;
    }
    .th-btn:hover {
        text-decoration: underline;
    }

    .detail-toolbar {
        display: flex;
        margin-bottom: 0.5rem;
    }
    .back-button {
        padding: 4px 8px;
        border: 1px solid #dee2e6;
        background: #fff;
        border-radius: 4px;
        cursor: pointer;
    }
    .back-button:hover {
        background: #f8f9fa;
    }
</style>
