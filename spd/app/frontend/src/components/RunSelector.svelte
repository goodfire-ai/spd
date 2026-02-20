<script lang="ts">
    import { onMount } from "svelte";
    import { discoverRuns, type DiscoveredRun } from "../lib/api/discover";
    import { fetchPretrainInfo, type PretrainInfoResponse } from "../lib/api/pretrainInfo";
    import { CANONICAL_RUNS, type RegistryEntry } from "../lib/registry";

    type Props = {
        onSelect: (wandbPath: string, contextLength: number) => void;
        isLoading: boolean;
        username: string | null;
    };

    let { onSelect, isLoading, username }: Props = $props();

    let customPath = $state("");
    let contextLength = $state(512);

    let discovered = $state<DiscoveredRun[] | "loading" | "error">("loading");
    let archInfo = $state<Record<string, PretrainInfoResponse | "loading" | "error">>({});

    const WANDB_PREFIX = "goodfire/spd";

    type MergedRun = {
        runId: string;
        wandbPath: string;
        registry: RegistryEntry | null;
        discovered: DiscoveredRun | null;
    };

    function extractRunId(wandbPath: string): string {
        const parts = wandbPath.split("/");
        return parts[parts.length - 1];
    }

    let mergedRuns = $derived.by((): MergedRun[] => {
        const discoveredMap = new Map<string, DiscoveredRun>();
        if (discovered !== "loading" && discovered !== "error") {
            for (const d of discovered) {
                discoveredMap.set(d.run_id, d);
            }
        }

        const seen = new Set<string>();
        const result: MergedRun[] = [];

        // Registry runs first (preserves ordering)
        for (const entry of CANONICAL_RUNS) {
            const runId = extractRunId(entry.wandbRunId);
            seen.add(runId);
            result.push({
                runId,
                wandbPath: entry.wandbRunId,
                registry: entry,
                discovered: discoveredMap.get(runId) ?? null,
            });
        }

        // Discovered-only runs after
        if (discovered !== "loading" && discovered !== "error") {
            for (const d of discovered) {
                if (!seen.has(d.run_id)) {
                    result.push({
                        runId: d.run_id,
                        wandbPath: `${WANDB_PREFIX}/${d.run_id}`,
                        registry: null,
                        discovered: d,
                    });
                }
            }
        }

        return result;
    });

    onMount(() => {
        // Fetch arch info for registry runs immediately
        for (const entry of CANONICAL_RUNS) {
            const runId = extractRunId(entry.wandbRunId);
            archInfo[runId] = "loading";
            fetchPretrainInfo(entry.wandbRunId).then(
                (info) => {
                    archInfo[runId] = info;
                },
                () => {
                    archInfo[runId] = "error";
                },
            );
        }

        // Discover runs from SPD_OUT_DIR
        discoverRuns().then(
            (runs) => {
                discovered = runs;
                // Fetch arch info for newly discovered runs (not already in registry)
                const registryIds = new Set(CANONICAL_RUNS.map((e) => extractRunId(e.wandbRunId)));
                for (const run of runs) {
                    if (!registryIds.has(run.run_id)) {
                        archInfo[run.run_id] = "loading";
                        fetchPretrainInfo(`${WANDB_PREFIX}/${run.run_id}`).then(
                            (info) => {
                                archInfo[run.run_id] = info;
                            },
                            () => {
                                archInfo[run.run_id] = "error";
                            },
                        );
                    }
                }
            },
            () => {
                discovered = "error";
            },
        );
    });

    function handleRunSelect(wandbPath: string) {
        onSelect(wandbPath, contextLength);
    }

    function handleCustomSubmit(event: Event) {
        event.preventDefault();
        const path = customPath.trim();
        if (!path) return;
        onSelect(path, contextLength);
    }

    type PillInfo = { label: string; present: boolean };

    function getDataPills(run: DiscoveredRun): PillInfo[] {
        return [
            { label: "harvest", present: run.has_harvest },
            { label: "det", present: run.has_detection },
            { label: "fuzz", present: run.has_fuzzing },
            { label: "intruder", present: run.has_intruder },
            { label: "ds attrs", present: run.has_dataset_attributions },
        ];
    }
</script>

<div class="selector-container">
    {#if isLoading}
        <div class="loading-overlay">
            <div class="spinner"></div>
            <p class="loading-text">Loading run...</p>
        </div>
    {/if}
    <div class="selector-content" class:dimmed={isLoading}>
        <h1 class="title">
            {#if username}
                Hello, {username}
            {:else}
                SPD Explorer
            {/if}
        </h1>

        <div class="runs-grid">
            {#each mergedRuns as run (run.runId)}
                {@const info = archInfo[run.runId]}
                <button
                    class="run-card"
                    onclick={() => handleRunSelect(run.wandbPath)}
                    disabled={isLoading}
                >
                    {#if run.registry}
                        <span class="run-model">{run.registry.modelName}</span>
                    {/if}
                    <span class="run-id">{run.runId}</span>
                    {#if run.registry?.notes}
                        <span class="run-notes">{run.registry.notes}</span>
                    {/if}
                    {#if info && info !== "loading" && info !== "error"}
                        <span class="run-arch">{info.summary}</span>
                    {:else if info === "loading"}
                        <span class="run-arch loading">...</span>
                    {/if}
                    {#if run.discovered}
                        <span class="run-labels">{run.discovered.n_labels} labels</span>
                        <div class="pills">
                            {#each getDataPills(run.discovered) as pill}
                                {#if pill.present}
                                    <span class="pill">{pill.label}</span>
                                {/if}
                            {/each}
                        </div>
                    {/if}
                    {#if run.registry?.clusterMappings}
                        <span class="run-cluster-mappings"
                            >{run.registry.clusterMappings.length} clustering runs</span
                        >
                    {/if}
                </button>
            {/each}
        </div>

        <div class="divider">
            <span>or enter a custom path</span>
        </div>

        <form class="custom-form" onsubmit={handleCustomSubmit}>
            <input
                type="text"
                placeholder="e.g. goodfire/spd/runs/33n6xjjt"
                bind:value={customPath}
                disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !customPath.trim()}>
                {isLoading ? "Loading..." : "Load"}
            </button>
        </form>
    </div>
</div>

<style>
    .selector-container {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: var(--bg-base);
        padding: var(--space-4);
    }

    .loading-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: var(--space-3);
        background: rgba(0, 0, 0, 0.5);
        z-index: 100;
    }

    .spinner {
        width: 40px;
        height: 40px;
        border: 3px solid var(--border-default);
        border-top-color: var(--accent-primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    .loading-text {
        color: var(--text-primary);
        font-family: var(--font-sans);
        font-size: var(--text-sm);
        margin: 0;
    }

    .selector-content {
        max-width: 900px;
        width: 100%;
        transition: opacity var(--transition-slow);
    }

    .selector-content.dimmed {
        opacity: 0.3;
        pointer-events: none;
    }

    .title {
        font-size: var(--text-3xl);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-4) 0;
        text-align: center;
        font-family: var(--font-sans);
    }

    .runs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: var(--space-3);
        margin-bottom: var(--space-6);
    }

    .run-card {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-1);
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        cursor: pointer;
        text-align: left;
        transition:
            border-color var(--transition-normal),
            background var(--transition-normal);
    }

    .run-card:hover:not(:disabled) {
        border-color: var(--accent-primary);
        background: var(--bg-elevated);
    }

    .run-card:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .run-model {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    .run-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
    }

    .run-notes {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .run-arch {
        font-size: 10px;
        font-family: var(--font-mono);
        color: var(--text-secondary, var(--text-muted));
        background: var(--bg-inset, var(--bg-base));
        padding: 1px 4px;
        border-radius: 3px;
        line-height: 1.3;
    }

    .run-arch.loading {
        opacity: 0.5;
        font-style: italic;
        font-family: var(--font-sans);
        background: none;
    }

    .run-labels {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .pills {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-top: 2px;
    }

    .pill {
        font-size: 10px;
        font-family: var(--font-sans);
        padding: 1px 6px;
        border-radius: 9999px;
        line-height: 1.4;
        white-space: nowrap;
        background: var(--accent-primary);
        color: white;
    }

    .run-cluster-mappings {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .divider {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        margin-bottom: var(--space-4);
    }

    .divider::before,
    .divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-default);
    }

    .divider span {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .custom-form {
        display: flex;
        gap: var(--space-2);
    }

    .custom-form input[type="text"] {
        flex: 1;
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .custom-form input[type="text"]::placeholder {
        color: var(--text-muted);
    }

    .custom-form input[type="text"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .custom-form button {
        padding: var(--space-2) var(--space-4);
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        font-weight: 500;
        cursor: pointer;
        font-family: var(--font-sans);
    }

    .custom-form button:hover:not(:disabled) {
        opacity: 0.9;
    }

    .custom-form button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }
</style>
