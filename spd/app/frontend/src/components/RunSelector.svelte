<script lang="ts">
    import { onMount } from "svelte";
    import { discoverRuns, type DiscoveredRun } from "../lib/api/discover";
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

    let mergedRuns = $derived.by((): MergedRun[] | null => {
        if (discovered === "loading" || discovered === "error") return null;

        const discoveredMap = new Map<string, DiscoveredRun>();
        for (const d of discovered) {
            discoveredMap.set(d.run_id, d);
        }

        const registryMap = new Map<string, RegistryEntry>();
        for (const entry of CANONICAL_RUNS) {
            registryMap.set(extractRunId(entry.wandbRunId), entry);
        }

        const seen = new Set<string>();
        const result: MergedRun[] = [];

        // Discovered runs first (sorted by recency from backend)
        for (const d of discovered) {
            seen.add(d.run_id);
            result.push({
                runId: d.run_id,
                wandbPath: `${WANDB_PREFIX}/${d.run_id}`,
                registry: registryMap.get(d.run_id) ?? null,
                discovered: d,
            });
        }

        // Registry-only runs after (no discovered data)
        for (const entry of CANONICAL_RUNS) {
            const runId = extractRunId(entry.wandbRunId);
            if (!seen.has(runId)) {
                result.push({
                    runId,
                    wandbPath: entry.wandbRunId,
                    registry: entry,
                    discovered: null,
                });
            }
        }

        return result;
    });

    onMount(() => {
        discoverRuns().then(
            (runs) => {
                discovered = runs;
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

    const DATA_PILLS = ["harvest", "det", "fuzz", "intruder", "ds attrs"] as const;

    function presentPills(run: DiscoveredRun): string[] {
        const flags: Record<string, boolean> = {
            harvest: run.has_harvest,
            det: run.has_detection,
            fuzz: run.has_fuzzing,
            intruder: run.has_intruder,
            "ds attrs": run.has_dataset_attributions,
        };
        return DATA_PILLS.filter((p) => flags[p]);
    }

    function formatDate(iso: string): string {
        const d = new Date(iso);
        return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
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

        <div class="runs-list">
            {#if mergedRuns === null}
                <div class="loading-skeleton">
                    {#each Array(12) as _}
                        <div class="skeleton-row">
                            <div class="skeleton-block id"></div>
                            <div class="skeleton-block model"></div>
                            <div class="skeleton-block data"></div>
                        </div>
                    {/each}
                </div>
            {:else}
                {#each mergedRuns as run (run.runId)}
                    {@const d = run.discovered}
                    {@const pills = d ? presentPills(d) : []}
                    {@const arch = d?.arch_summary ?? null}
                    <button
                        class="run-row"
                        onclick={() => handleRunSelect(run.wandbPath)}
                        disabled={isLoading}
                    >
                        <span class="col-id">
                            <span class="run-id">{run.runId}</span>
                            <span class="run-notes"
                                >{run.registry?.notes ??
                                    (d?.created_at ? formatDate(d.created_at) : "")}</span
                            >
                        </span>
                        <span class="col-model">
                            {#if arch}
                                <span class="run-arch">{arch}</span>
                            {/if}
                        </span>
                        <span class="col-data">
                            {#if d}
                                <span class="label-count"
                                    >{d.n_labels.toLocaleString()} labels</span
                                >
                            {/if}
                            {#if pills.length > 0}
                                <span class="pills">
                                    {#each pills as pill}
                                        <span class="pill">{pill}</span>
                                    {/each}
                                </span>
                            {/if}
                        </span>
                    </button>
                {/each}
            {/if}
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
        max-width: 960px;
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

    .runs-list {
        height: 480px;
        overflow-y: auto;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        margin-bottom: var(--space-6);
    }

    /* Loading skeleton */
    .loading-skeleton {
        display: flex;
        flex-direction: column;
    }

    .skeleton-row {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: 10px var(--space-2);
    }

    .skeleton-block {
        height: 10px;
        border-radius: 4px;
        background: var(--border-default);
        animation: pulse 1.5s ease-in-out infinite;
    }

    .skeleton-block.id {
        width: 80px;
        flex-shrink: 0;
    }

    .skeleton-block.model {
        flex: 1;
    }

    .skeleton-block.data {
        width: 100px;
        flex-shrink: 0;
    }

    @keyframes pulse {
        0%,
        100% {
            opacity: 0.2;
        }
        50% {
            opacity: 0.45;
        }
    }

    /* Run rows */
    .run-row {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: 7px var(--space-2);
        width: 100%;
        background: transparent;
        border: none;
        border-radius: var(--radius-sm);
        cursor: pointer;
        text-align: left;
        transition: background var(--transition-normal);
    }

    .run-row:hover:not(:disabled) {
        background: var(--bg-elevated);
    }

    .run-row:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .col-id {
        display: flex;
        flex-direction: column;
        gap: 1px;
        flex-shrink: 0;
        width: 140px;
    }

    .run-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        font-weight: 500;
    }

    .run-notes {
        font-size: 10px;
        color: var(--text-muted);
        font-family: var(--font-sans);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.2;
    }

    .col-model {
        display: flex;
        flex-direction: column;
        gap: 1px;
        flex: 1;
        min-width: 0;
    }

    .run-arch {
        font-size: 10px;
        font-family: var(--font-mono);
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.3;
    }

    .col-data {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-shrink: 0;
    }

    .label-count {
        font-size: 11px;
        color: var(--text-muted);
        font-family: var(--font-mono);
        white-space: nowrap;
    }

    .pills {
        display: flex;
        gap: 3px;
    }

    .pill {
        font-size: 9px;
        font-family: var(--font-sans);
        font-weight: 500;
        padding: 1px 6px;
        border-radius: 9999px;
        line-height: 1.4;
        white-space: nowrap;
        background: color-mix(in srgb, var(--accent-primary) 15%, transparent);
        color: var(--accent-primary);
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
