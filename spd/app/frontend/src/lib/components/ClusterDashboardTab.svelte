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
    import VirtualList from "$lib/components/VirtualList.svelte";

    type TopToken = {
        token: string;
        count: number;
    };

    type ClusterRow = {
        id: number;
        clusterHash: string;
        componentCount: number;
        topTokens: TopToken[];
        maxTokenCount: number;
    };

    type ClusterExample = {
        textHash: string;
        rawText: string;
        offsetMapping: [number, number][];
        activations: number[];
    };

    type ClusterMap = Record<string, ClusterDataDTO>;

    let loading = true;
    let errorMsg: string | null = null;
    let rows: ClusterRow[] = [];
    let clusterMap: ClusterMap = {};
    let dashboard: ClusterDashboardResponse | null = null;
    let textSampleLookup: Record<string, { full_text: string; tokens: string[] }> = {};

    let iteration = 3000;
    let nSamples = 16;
    let nBatches = 2;
    let batchSize = 64;
    let contextLength = 64;
    let clusteringRunId = "goodfire/spd-cluster/j8dgvemf";

    let showDetail = false;
    let currentCluster: ClusterDataDTO | null = null;
    let sortKey: "id" | "componentCount" = "id";
    let sortDir: "asc" | "desc" = "asc";

    let pendingController: AbortController | null = null;
    let currentExamples: ClusterExample[] = [];

    const ROW_HEIGHT = 220;
    const VIRTUAL_BUFFER = 6;

    function tokenColor(count: number, max: number): string {
        if (max <= 0) return "transparent";
        const ratio = Math.min(Math.max(count / max, 0), 1);
        const opacity = 0.15 + ratio * 0.35;
        return `rgba(0, 200, 0, ${opacity})`;
    }

    function buildRows(clusters: ClusterDataDTO[]): ClusterRow[] {
        return clusters.map((cluster, idx) => {
            const parts = cluster.cluster_hash.split("-");
            const maybeId = Number.parseInt(parts[parts.length - 1] ?? "", 10);
            const id = Number.isNaN(maybeId) ? idx : maybeId;
            const topTokens: TopToken[] = (cluster.stats?.token_activations?.top_tokens ?? [])
                .slice(0, 5)
                .map((entry: any) => ({
                    token: typeof entry.token === "string" ? entry.token : "",
                    count: typeof entry.count === "number" ? entry.count : 0
                }));
            const maxTokenCount =
                topTokens.reduce((max, token) => Math.max(max, token.count), 0) || 1;
            return {
                id,
                clusterHash: cluster.cluster_hash,
                componentCount: cluster.components?.length ?? 0,
                topTokens,
                maxTokenCount
            } satisfies ClusterRow;
        });
    }

    function applySort(nextRows: ClusterRow[]): ClusterRow[] {
        const sorted = [...nextRows];
        if (sortKey === "id") {
            sorted.sort((a, b) => (sortDir === "asc" ? a.id - b.id : b.id - a.id));
        } else {
            sorted.sort((a, b) =>
                sortDir === "asc"
                    ? a.componentCount - b.componentCount
                    : b.componentCount - a.componentCount
            );
        }
        return sorted;
    }

    function buildOffsets(tokens: string[]): [number, number][] {
        const offsets: [number, number][] = [];
        let cursor = 0;
        for (const token of tokens) {
            const start = cursor;
            const end = cursor + token.length;
            offsets.push([start, end]);
            cursor = end;
        }
        return offsets;
    }

    function normalizeActivations(values: number[], length: number): number[] {
        if (!values.length) return new Array(length).fill(0);
        const slice = values.slice(0, length);
        const max = Math.max(...slice.map((v) => Math.max(v, 0)), 1e-6);
        return slice.map((v) => Math.max(v, 0) / max);
    }

    function buildExamples(cluster: ClusterDataDTO): ClusterExample[] {
        if (!dashboard) return [];
        const hashes = cluster.criterion_samples?.["max_activation-max-16"] ?? [];
        const activations = dashboard.activation_batch.activations ?? [];
        const tokensList = dashboard.activation_batch.tokens ?? [];
        const examples: ClusterExample[] = [];

        for (const textHash of hashes.slice(0, 5)) {
            const activationHash = `${cluster.cluster_hash}:${textHash}`;
            const idx = dashboard.activations_map[activationHash];
            if (typeof idx !== "number") continue;

            const tokens = tokensList[idx] ?? textSampleLookup[textHash]?.tokens ?? [];
            const activationValues = activations[idx] ?? [];
            if (!tokens.length || !activationValues.length) continue;

            const rawText = tokens.join("");
            const offsets = buildOffsets(tokens);
            const normalized = normalizeActivations(activationValues, offsets.length);

            examples.push({
                textHash,
                rawText,
                offsetMapping: offsets,
                activations: normalized
            });
        }

        return examples;
    }

    function resetDetail() {
        showDetail = false;
        currentCluster = null;
        currentExamples = [];
    }

    async function fetchDashboard() {
        pendingController?.abort();
        const controller = new AbortController();
        pendingController = controller;

        loading = true;
        errorMsg = null;

        try {
            console.log("fetching dashboard");
            const result = await getClusterDashboardData({
                iteration,
                n_samples: nSamples,
                n_batches: nBatches,
                batch_size: batchSize,
                context_length: contextLength,
                clustering_run: clusteringRunId,
                signal: controller.signal
            });
            console.log("dashboard fetched");
            if (controller.signal.aborted) return;

            dashboard = result;
            clusterMap = Object.fromEntries(
                result.clusters.map((cluster) => [cluster.cluster_hash, cluster])
            );
            textSampleLookup = Object.fromEntries(
                result.text_samples.map((sample) => [
                    sample.text_hash,
                    {
                        full_text: sample.full_text,
                        tokens: sample.tokens
                    }
                ])
            );
            console.log("clusterMap built, sorting rows");
            const _rows = buildRows(result.clusters);
            rows = applySort(_rows);
            console.log("rows sorted");

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
        currentExamples = buildExamples(cluster);
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
        ? ([
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
          ] as const)
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
            <div class="list">
                <div class="list-header">
                    <div class="cell col-id">
                        <button class="th-btn" on:click={() => toggleSort("id")}>ID</button>
                    </div>
                    <div class="cell col-comps">
                        <button class="th-btn" on:click={() => toggleSort("componentCount")}
                            >Comps</button
                        >
                    </div>
                    <div class="cell col-model">Model View</div>
                    <div class="cell col-hist">All Activations</div>
                    <div class="cell col-hist">Max Activation Pos</div>
                    <div class="cell col-tokens">Top Tokens</div>
                    <div class="cell col-actions">Actions</div>
                </div>
                <VirtualList
                    items={rows}
                    rowHeight={ROW_HEIGHT}
                    buffer={VIRTUAL_BUFFER}
                    getKey={(row: ClusterRow) => row.clusterHash}
                    let:item
                >
                    <div class="list-row">
                        <div class="cell col-id">{item.id}</div>
                        <div class="cell col-comps">{item.componentCount}</div>
                        <div class="cell col-model">
                            <MiniModelView
                                components={clusterMap[item.clusterHash].components ?? []}
                            />
                        </div>
                        <div class="cell col-hist">
                            {#if clusterMap[item.clusterHash].stats?.all_activations?.bin_counts}
                                <Sparkbars
                                    bins={clusterMap[item.clusterHash]!.stats!.all_activations!
                                        .bin_counts as number[]}
                                    width={160}
                                    height={48}
                                    color="#4169E1"
                                />
                            {:else}
                                <div class="hist-placeholder"></div>
                            {/if}
                        </div>
                        <div class="cell col-hist">
                            {#if clusterMap[item.clusterHash].stats?.max_activation_position?.bin_counts}
                                <Sparkbars
                                    bins={clusterMap[item.clusterHash]!.stats!
                                        .max_activation_position!.bin_counts as number[]}
                                    width={160}
                                    height={48}
                                    color="#DC143C"
                                />
                            {:else}
                                <div class="hist-placeholder"></div>
                            {/if}
                        </div>
                        <div class="cell col-tokens">
                            {#if item.topTokens.length}
                                <ul class="token-list">
                                    {#each item.topTokens as token, idx (idx)}
                                        <li
                                            style={`background:${tokenColor(token.count, item.maxTokenCount)};`}
                                        >
                                            <code>{token.token.replace(/\s/g, "·")}</code>
                                            <span class="token-count">({token.count})</span>
                                        </li>
                                    {/each}
                                </ul>
                            {:else}
                                <span class="no-tokens">—</span>
                            {/if}
                        </div>
                        <div class="cell col-actions">
                            <button class="view-button" on:click={() => onView(item)}>View →</button
                            >
                        </div>
                    </div>
                </VirtualList>
            </div>
        </div>
    {:else}
        <div class="detail-toolbar">
            <button class="back-button" on:click={resetDetail}>← Back</button>
        </div>
        <ClusterDetailNew cluster={currentCluster} examples={currentExamples} />
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
        display: flex;
        flex-direction: column;
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

    .list {
        border: 1px solid #dee2e6;
        border-radius: 6px;
        overflow: hidden;
        background: #fff;
    }

    .list :global(.virtual-viewport) {
        max-height: 60vh;
        overflow-y: auto;
    }

    .list :global(.virtual-content) {
        width: 100%;
    }

    .list-header,
    .list-row {
        display: grid;
        grid-template-columns: 70px 110px 220px 220px 220px 240px 120px;
        column-gap: 1rem;
        align-items: stretch;
        padding: 0.75rem 1rem;
    }

    .list-header {
        background: #f8f9fa;
        border-bottom: 1px solid #dee2e6;
        font-weight: 600;
    }

    .list-row {
        min-height: 200px;
        border-bottom: 1px solid #f1f3f5;
    }

    .list-row:hover {
        background: #f9fafc;
    }

    .cell {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #333;
    }

    .cell.col-id {
        font-weight: 600;
    }

    .cell.col-comps {
        justify-content: flex-end;
        font-variant-numeric: tabular-nums;
    }

    .cell.col-actions {
        justify-content: flex-end;
    }

    .cell.col-model {
        min-height: 160px;
    }

    .col-tokens {
        min-width: 200px;
    }

    .token-list {
        margin: 0;
        padding-left: 0;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }

    .token-list li {
        list-style: none;
        color: #333;
        padding: 2px 6px;
        border-radius: 4px;
        font-variant-numeric: tabular-nums;
    }

    .token-list code {
        font-family:
            ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New",
            monospace;
        font-size: 12px;
    }

    .token-count {
        margin-left: 0.35rem;
        color: #666;
        font-size: 12px;
    }

    .no-tokens {
        color: #999;
    }

    .col-hist {
        min-width: 200px;
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
