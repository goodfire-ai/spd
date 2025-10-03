<script lang="ts">
    import Sparkbars from "$lib/components/Sparkbars.svelte";

    type ClusterStats = Record<string, any>;
    type ClusterDataLocal = {
        cluster_hash: string;
        components: { module: string; index: number; label?: string }[];
        stats?: ClusterStats;
        criterion_samples?: Record<string, string[]>;
    };

    export let cluster: ClusterDataLocal | null = null;

    $: histogramStats = (() => {
        if (!cluster?.stats) return [] as string[];
        const keys: string[] = [];
        for (const [k, v] of Object.entries(cluster.stats)) {
            if (v && typeof v === "object" && "bin_counts" in (v as any) && "bin_edges" in (v as any)) {
                keys.push(k);
            }
        }
        return keys;
    })();

    const statColors: Record<string, string> = {
        'all_activations': '#4169E1',
        'max_activation-max-16': '#DC143C',
        'max_activation-max-32': '#DC143C',
        'mean_activation-max-16': '#228B22',
        'median_activation-max-16': '#FF8C00',
        'min_activation-max-16': '#9370DB',
        'max_activation_position': '#FF6347'
    };
</script>

{#if !cluster}
    <div class="status">No cluster selected.</div>
{:else}
    <div class="detail-container">
        <div class="header">
            <h3>Cluster {cluster.cluster_hash}</h3>
            <div class="meta">Components: {cluster.components.length}</div>
        </div>

        <div class="section">
            <h4>Components</h4>
            <table class="components-table">
                <thead>
                    <tr>
                        <th>Module</th>
                        <th class="col-right">Index</th>
                    </tr>
                </thead>
                <tbody>
                    {#each cluster.components as c}
                        <tr>
                            <td>{c.module}</td>
                            <td class="col-right">{c.index}</td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>

        {#if histogramStats.length}
            <div class="section">
                <h4>Histograms</h4>
                <div class="hist-grid">
                    {#each histogramStats as k}
                        <div class="hist-item">
                            <div class="hist-label">{k}</div>
                            <Sparkbars bins={(cluster?.stats?.[k]?.bin_counts) ?? []} color={statColors[k] ?? '#808080'} />
                        </div>
                    {/each}
                </div>
            </div>
        {/if}

        {#if cluster.stats?.token_activations}
            <div class="section">
                <h4>Token Activations</h4>
                <div class="token-stats">
                    <div>Unique: {cluster.stats.token_activations.total_unique_tokens}</div>
                    <div>Total: {cluster.stats.token_activations.total_activations}</div>
                    <div>Entropy: {cluster.stats.token_activations.entropy?.toFixed?.(2)}</div>
                    <div>Conc: {(cluster.stats.token_activations.concentration_ratio * 100).toFixed(1)}%</div>
                </div>
                {#if cluster.stats.token_activations.top_tokens?.length}
                    <table class="components-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Token</th>
                                <th class="col-right">Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each cluster.stats.token_activations.top_tokens.slice(0, 10) as item, i}
                                <tr>
                                    <td>{i + 1}</td>
                                    <td><code class="token-code">{item.token.replace(/ /g, '·').replace(/\n/g, '↵')}</code></td>
                                    <td class="col-right">{item.count}</td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                {/if}
            </div>
        {/if}

        <!-- Samples/activations: add after NDArray loader abstraction is ready -->
    </div>
{/if}

<style>
    .status {
        color: #333;
    }
    .detail-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .header {
        display: flex;
        align-items: baseline;
        gap: 1rem;
    }
    .meta {
        color: #6c757d;
    }
    .section h4 {
        margin: 0 0 0.5rem 0;
    }
    .components-table {
        width: 100%;
        border-collapse: collapse;
    }
    .components-table th,
    .components-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
        text-align: left;
    }
    .col-right {
        text-align: right;
    }
    .hist-grid {
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    }
    .hist-item {
        background: #fff;
        border: 1px solid #eee;
        border-radius: 6px;
        padding: 8px;
    }
    .hist-label {
        font-size: 12px;
        font-weight: 600;
        color: #555;
        margin-bottom: 6px;
    }
    .token-stats {
        display: flex;
        gap: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .token-code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
    }
</style>

