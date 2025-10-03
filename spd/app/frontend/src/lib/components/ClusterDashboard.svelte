<script lang="ts">
    import { onMount } from "svelte";
    import { API_URL, getClusterDashboardDataDirs, type ClusterDashboardDataDirs } from "$lib/api";

    export let runId: string | null = null;

    let dataDirs: ClusterDashboardDataDirs | null = null;
    let absDirs: string[] = [];
    let selectedDir: string | null = null;
    let scriptsReady = false;
    let initializing = true;
    let lastRunId: string | null = null;

    const assetBase = `${API_URL}/cluster-dashboard`;

    function installConfigFetchShim() {
        const origFetch = window.fetch.bind(window);
        // Only rewrite the dashboard's config.json request
        window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
            try {
                if (typeof input === "string") {
                    if (input === "config.json") {
                        return origFetch(`${assetBase}/config.json`, init);
                    }
                }
            } catch (_) {
                // fall through
            }
            return origFetch(input as any, init);
        };
    }

    function loadScript(src: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = src;
            s.async = false; // preserve order
            s.onload = () => resolve();
            s.onerror = () => reject(new Error(`Failed to load script: ${src}`));
            document.head.appendChild(s);
        });
    }

    async function ensureDashboardScriptsLoaded() {
        if (scriptsReady) return;
        // Ensure config.json requests resolve to backend static path
        installConfigFetchShim();
        const scripts = [
            `${assetBase}/js/util/notif.js`,
            `${assetBase}/js/util/config.js`,
            `${assetBase}/js/util/ColorUtil.js`,
            `${assetBase}/js/util/sparklines.js`,
            `${assetBase}/js/pkg/jszip.min.js`,
            `${assetBase}/js/util/table.js`,
            `${assetBase}/js/model-visualization.js`,
            `${assetBase}/js/cluster-selection.js`
        ];
        for (const src of scripts) {
            await loadScript(src);
        }
        scriptsReady = true;
    }

    function clearDashboardContainers() {
        const container = document.getElementById("clusterTableContainer");
        if (container) container.innerHTML = "";
        const loading = document.getElementById("loading");
        if (loading) loading.style.display = "block";
    }

    async function loadDashboard(dir: string) {
        await ensureDashboardScriptsLoaded();
        // @ts-expect-error globals from legacy dashboard
        if (typeof initConfig === "function") {
            // only initialize config on first load
            if (initializing) {
                // @ts-expect-error global
                await initConfig();
            }
            // @ts-expect-error global
            if (typeof setConfigValue === "function") {
                // Avoid URL churn by not updating the URL
                // @ts-expect-error global
                setConfigValue("data.dataDir", dir, false);
            }
        }
        clearDashboardContainers();
        // @ts-expect-error global
        if (typeof loadData === "function") {
            // @ts-expect-error global
            loadData();
        }
        initializing = false;
    }

    async function refreshDirs() {
        await ensureDashboardScriptsLoaded();
        dataDirs = await getClusterDashboardDataDirs(runId ?? undefined);
        absDirs = (dataDirs?.dirs ?? []).map((d) => `${assetBase}/${d}`);
        const latestRel = dataDirs?.latest ?? dataDirs?.dirs?.[0] ?? null;
        selectedDir = latestRel ? `${assetBase}/${latestRel}` : null;
        if (selectedDir) await loadDashboard(selectedDir);
    }

    onMount(() => {
        refreshDirs().catch((e) => console.error(e));
    });

    $: if (scriptsReady && runId !== lastRunId) {
        lastRunId = runId;
        refreshDirs().catch((e) => console.error(e));
    }

    function onSelectChange(e: Event) {
        const target = e.target as HTMLSelectElement;
        selectedDir = target.value || null;
        if (selectedDir) {
            loadDashboard(selectedDir).catch((e) => console.error(e));
        }
    }
</script>

<svelte:head>
    <!-- <title>Cluster Selection</title> -->
    <link rel="stylesheet" href={`${assetBase}/css/styles.css`}>
    <link rel="stylesheet" href={`${assetBase}/css/notif.css`}>
    <link rel="stylesheet" href={`${assetBase}/css/model-view.css`}>
</svelte:head>

<div style="display: flex; flex-direction: column; gap: 0.75rem;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="margin-left: auto; display: flex; align-items: center; gap: 0.5rem;">
            <label for="data-dir" style="font-size: 0.9rem; color: #555;">Data dir</label>
            <select id="data-dir" on:change={onSelectChange} disabled={!absDirs.length}>
                {#if absDirs.length}
                    {#each absDirs as d}
                        <option value={d} selected={d === selectedDir}>{d.replace(`${assetBase}/`, "")}</option>
                    {/each}
                {:else}
                    <option value="">No data directories found</option>
                {/if}
            </select>
        </div>
    </div>

    <div id="modelInfo" style="margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; display: none;">
        <h2 style="margin-top: 0;">Model Information</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div>
                <strong>Total Modules:</strong> <span id="totalModules">-</span>
            </div>
            <div>
                <strong>Total Components:</strong> <span id="totalComponents">-</span>
            </div>
            <div>
                <strong>Total Clusters:</strong> <span id="totalClusters">-</span>
            </div>
            <div>
                <strong>Model Parameters:</strong> <span id="totalParameters">-</span>
            </div>
        </div>
    </div>

    <div id="clusterTableContainer"></div>
    <div id="loading">Loading data...</div>
    <div id="tooltip"></div>
</div>

<style>
</style>


