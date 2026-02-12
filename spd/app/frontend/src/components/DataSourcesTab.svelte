<script lang="ts">
    import { getContext, onMount } from "svelte";
    import type { Loadable } from "../lib";
    import { fetchDataSources, type DataSourcesResponse } from "../lib/api";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    let data = $state<Loadable<DataSourcesResponse>>({ status: "uninitialized" });

    onMount(async () => {
        data = { status: "loading" };
        try {
            const result = await fetchDataSources();
            data = { status: "loaded", data: result };
        } catch (e) {
            data = { status: "error", error: e };
        }
    });

    function formatConfigValue(value: unknown): string {
        if (value === null || value === undefined) return "—";
        if (typeof value === "object") return JSON.stringify(value);
        return String(value);
    }
</script>

<div class="data-sources-container">
    {#if runState.run.status === "loaded" && runState.run.data.config_yaml}
        <section class="source-section">
            <h3 class="section-title">Run Config</h3>
            <pre class="config-yaml">{runState.run.data.config_yaml}</pre>
        </section>
    {/if}

    {#if data.status === "loading"}
        <p class="status-text">Loading data sources...</p>
    {:else if data.status === "error"}
        <p class="status-text error">Failed to load data sources: {data.error}</p>
    {:else if data.status === "loaded"}
        {@const { harvest, autointerp, attributions } = data.data}

        {#if !harvest && !autointerp && !attributions}
            <p class="status-text">No pipeline data available for this run.</p>
        {/if}

        {#if harvest}
            <section class="source-section">
                <h3 class="section-title">Harvest</h3>
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{harvest.subrun_id}</span>

                    <span class="label">Components</span>
                    <span class="value">{harvest.n_components.toLocaleString()}</span>

                    <span class="label">Intruder eval</span>
                    <span class="value">{harvest.has_intruder_scores ? "yes" : "no"}</span>

                    {#each Object.entries(harvest.config) as [key, value] (key)}
                        <span class="label">{key}</span>
                        <span class="value mono">{formatConfigValue(value)}</span>
                    {/each}
                </div>
            </section>
        {/if}

        {#if attributions}
            <section class="source-section">
                <h3 class="section-title">Dataset Attributions</h3>
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{attributions.subrun_id}</span>

                    <span class="label">Batches</span>
                    <span class="value">{attributions.n_batches_processed.toLocaleString()}</span>

                    <span class="label">Tokens</span>
                    <span class="value">{attributions.n_tokens_processed.toLocaleString()}</span>

                    <span class="label">CI threshold</span>
                    <span class="value mono">{attributions.ci_threshold}</span>
                </div>
            </section>
        {/if}

        {#if autointerp}
            <section class="source-section">
                <h3 class="section-title">Autointerp</h3>
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{autointerp.subrun_id}</span>

                    <span class="label">Interpretations</span>
                    <span class="value">{autointerp.n_interpretations.toLocaleString()}</span>

                    <span class="label">Eval scores</span>
                    <span class="value">
                        {#if autointerp.eval_scores.length > 0}
                            {autointerp.eval_scores.join(", ")}
                        {:else}
                            <span class="muted">none</span>
                        {/if}
                    </span>

                    {#each Object.entries(autointerp.config) as [key, value] (key)}
                        <span class="label">{key}</span>
                        <span class="value mono">{formatConfigValue(value)}</span>
                    {/each}
                </div>
            </section>
        {/if}
    {/if}
</div>

<style>
    .data-sources-container {
        padding: var(--space-6);
        display: flex;
        flex-direction: column;
        gap: var(--space-6);
        max-width: 640px;
    }

    .status-text {
        color: var(--text-muted);
        font-family: var(--font-sans);
        font-size: var(--text-sm);
    }

    .status-text.error {
        color: var(--accent-primary);
    }

    .source-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .section-title {
        font-family: var(--font-sans);
        font-size: var(--text-base);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-default);
    }

    .info-grid {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: var(--space-1) var(--space-4);
        font-size: var(--text-sm);
    }

    .label {
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .value {
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    .value.mono {
        font-family: var(--font-mono);
    }

    .muted {
        color: var(--text-muted);
    }

    .config-yaml {
        max-height: 50vh;
        overflow: auto;
        margin: 0;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-primary);
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
