<script lang="ts">
    import { onMount } from "svelte";
    import { API_URL, getClusterDashboardDataDirs, type ClusterDashboardDataDirs } from "$lib/api";

    type ClusterRow = {
        id: number;
        clusterHash: string;
        componentCount: number;
        modules: string[];
    };

    export let runId: string | null = null;
    export let demo: boolean = true;
    let dataDirs: ClusterDashboardDataDirs | null = null;
    let selectedDirRel: string | null = null; // relative under /cluster-dashboard
    let selectedDirAbs: string | null = null; // absolute URL for fetch
    let loading = true;
    let errorMsg: string | null = null;
    let rows: ClusterRow[] = [];
    let clusterMap: Record<string, any> = {};

    const assetBase = `${API_URL}/cluster-dashboard`;

    async function fetchText(url: string): Promise<string> {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.text();
    }

    async function loadClusters(dirAbs: string) {
        rows = [];
        errorMsg = null;
        loading = true;
        try {
            const clustersUrl = `${dirAbs}/clusters.jsonl`;
            const text = await fetchText(clustersUrl);
            const byHash: Record<string, any> = {};
            for (const line of text.trim().split("\n")) {
                if (!line) continue;
                const obj = JSON.parse(line);
                byHash[obj.cluster_hash] = obj;
            }

            const nextRows: ClusterRow[] = [];
            for (const [clusterHash, cluster] of Object.entries(byHash)) {
                const modules = new Set<string>();
                (cluster as any).components.forEach((c: any) => modules.add(c.module));
                const parts = clusterHash.split("-");
                const id = parseInt(parts[parts.length - 1]);
                nextRows.push({
                    id,
                    clusterHash,
                    componentCount: (cluster as any).components.length,
                    modules: Array.from(modules)
                });
            }
            nextRows.sort((a, b) => a.id - b.id);
            rows = nextRows;
            clusterMap = byHash;
        } catch (e: any) {
            errorMsg = e?.message ?? String(e);
        }
        loading = false;
    }

    import { getDemoClusterRows, getDemoClusterData } from "$lib/api";
    import ClusterDetailNew from "$lib/components/ClusterDetailNew.svelte";
    import MiniModelView from "$lib/components/MiniModelView.svelte";
    import Sparkbars from "$lib/components/Sparkbars.svelte";

    async function init() {
        loading = true;
        errorMsg = null;
        try {
            if (demo) {
                // Populate demo rows immediately
                rows = getDemoClusterRows(24);
                dataDirs = { dirs: ["data/demo"], latest: "data/demo" };
                selectedDirRel = "data/demo";
                selectedDirAbs = `${assetBase}/data/demo`;
                const map: Record<string, any> = {};
                for (const r of rows) {
                    map[r.clusterHash] = getDemoClusterData(r.clusterHash);
                }
                clusterMap = map;
                loading = false;
                return;
            }
            dataDirs = await getClusterDashboardDataDirs(runId ?? undefined);
            const rel = dataDirs?.latest ?? dataDirs?.dirs?.[0] ?? null;
            selectedDirRel = rel;
            selectedDirAbs = rel ? `${assetBase}/${rel}` : null;
            if (selectedDirAbs) {
                await loadClusters(selectedDirAbs);
            } else {
                loading = false;
            }
        } catch (e: any) {
            errorMsg = e?.message ?? String(e);
            loading = false;
        }
    }

    onMount(() => {
        init().catch((e) => (errorMsg = e?.message ?? String(e)));
    });

    function onDirChange(e: Event) {
        const v = (e.target as HTMLSelectElement).value || null;
        selectedDirRel = v;
        selectedDirAbs = v ? `${assetBase}/${v}` : null;
        if (selectedDirAbs)
            loadClusters(selectedDirAbs).catch((e) => (errorMsg = e?.message ?? String(e)));
    }

    let showDetail = false;
    let currentCluster: any | null = null;
    let sortKey: "id" | "componentCount" = "id";
    let sortDir: "asc" | "desc" = "asc";

    let modelInfoDemo = {
        totalModules: 28,
        totalComponents: 3241,
        totalClusters: 2240,
        totalParameters: "127.5M"
    };

    async function loadClusterObject(dirAbs: string, clusterHash: string) {
        if (demo) return getDemoClusterData(clusterHash);
        const clustersUrl = `${dirAbs}/clusters.jsonl`;
        const text = await fetchText(clustersUrl);
        for (const line of text.trim().split("\n")) {
            if (!line) continue;
            const obj = JSON.parse(line);
            if (obj.cluster_hash === clusterHash) return obj;
        }
        return null;
    }

    async function onView(row: ClusterRow) {
        if (!selectedDirAbs) return;
        currentCluster = clusterMap[row.clusterHash] ?? await loadClusterObject(selectedDirAbs, row.clusterHash);
        showDetail = true;
    }
</script>

<div class="dashboard-container">
    <div class="toolbar">
        <div class="toolbar-right">
            <label for="data-dir" class="toolbar-label">Data dir</label>
            <select
                id="data-dir"
                on:change={onDirChange}
                disabled={!dataDirs || dataDirs.dirs.length === 0}
            >
                {#if dataDirs?.dirs?.length}
                    {#each dataDirs.dirs as d}
                        <option value={d} selected={d === selectedDirRel}>{d}</option>
                    {/each}
                {:else}
                    <option value="">No data directories found</option>
                {/if}
            </select>
        </div>
    </div>

    {#if loading}
        <div class="status">Loading...</div>
    {:else if errorMsg}
        <div class="status-error">{errorMsg}</div>
    {:else if !showDetail}
        <div class="table-wrapper">
            <div class="model-info">
                <div>Total Modules: {modelInfoDemo.totalModules}</div>
                <div>Total Components: {modelInfoDemo.totalComponents}</div>
                <div>Total Clusters: {modelInfoDemo.totalClusters}</div>
                <div>Model Parameters: {modelInfoDemo.totalParameters}</div>
            </div>
            <table class="cluster-table">
                <thead>
                    <tr>
                        <th class="col-id">
                            <button
                                class="th-btn"
                                on:click={() => {
                                    const same = sortKey === "id";
                                    sortKey = "id";
                                    sortDir = same && sortDir === "asc" ? "desc" : "asc";
                                    rows = [...rows].sort((a, b) =>
                                        sortDir === "asc" ? a.id - b.id : b.id - a.id
                                    );
                                }}>ID</button
                            >
                        </th>
                        <th class="col-comps">
                            <button
                                class="th-btn"
                                on:click={() => {
                                    const same = sortKey === "componentCount";
                                    sortKey = "componentCount";
                                    sortDir = same && sortDir === "asc" ? "desc" : "asc";
                                    rows = [...rows].sort((a, b) =>
                                        sortDir === "asc"
                                            ? a.componentCount - b.componentCount
                                            : b.componentCount - a.componentCount
                                    );
                                }}>Comps</button
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
                                    components={(clusterMap[row.clusterHash]?.components) ?? []}
                                    maxLayers={12}
                                />
                            </td>
                            <td class="col-hist">
                                {#if clusterMap[row.clusterHash]?.stats?.all_activations?.bin_counts}
                                    <Sparkbars
                                        bins={clusterMap[row.clusterHash].stats.all_activations.bin_counts}
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
                                        bins={clusterMap[row.clusterHash].stats.max_activation_position.bin_counts}
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
            <button class="back-button" on:click={() => (showDetail = false)}>← Back</button>
        </div>
        <ClusterDetailNew cluster={currentCluster} />
    {/if}
</div>

<style>
    .dashboard-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .toolbar {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .toolbar-right {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .toolbar-label {
        font-size: 0.9rem;
        color: #555;
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
