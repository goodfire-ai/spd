<script lang="ts">
    import { onMount } from "svelte";
    import { CANONICAL_RUNS, formatRunIdForDisplay, type RegistryEntry } from "../lib/registry";
    import { fetchPretrainInfo, type PretrainInfoResponse } from "../lib/api/pretrainInfo";

    type Props = {
        onSelect: (wandbPath: string, contextLength: number) => void;
        isLoading: boolean;
        username: string | null;
    };

    let { onSelect, isLoading, username }: Props = $props();

    let customPath = $state("");
    let contextLength = $state(512);

    // Architecture info fetched in real-time for each canonical run
    let archInfo = $state<Record<string, PretrainInfoResponse | "loading" | "error">>({});

    onMount(() => {
        for (const entry of CANONICAL_RUNS) {
            archInfo[entry.wandbRunId] = "loading";
            fetchPretrainInfo(entry.wandbRunId).then(
                (info) => {
                    archInfo[entry.wandbRunId] = info;
                },
                () => {
                    archInfo[entry.wandbRunId] = "error";
                },
            );
        }
    });

    function handleRegistrySelect(entry: RegistryEntry) {
        onSelect(entry.wandbRunId, contextLength);
    }

    function handleCustomSubmit(event: Event) {
        event.preventDefault();
        const path = customPath.trim();
        if (!path) return;
        onSelect(path, contextLength);
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
            {#each CANONICAL_RUNS as entry (entry.wandbRunId)}
                {@const info = archInfo[entry.wandbRunId]}
                <button class="run-card" onclick={() => handleRegistrySelect(entry)} disabled={isLoading}>
                    <span class="run-model">{entry.modelName}</span>
                    <span class="run-id">{formatRunIdForDisplay(entry.wandbRunId)}</span>
                    {#if entry.notes}
                        <span class="run-notes">{entry.notes}</span>
                    {/if}
                    {#if info && info !== "loading" && info !== "error"}
                        <span class="run-arch">{info.summary}</span>
                    {:else if info === "loading"}
                        <span class="run-arch loading">loading arch...</span>
                    {/if}
                    {#if entry.clusterMappings}
                        <span class="run-cluster-mappings">{entry.clusterMappings.length} clustering runs</span>
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
        max-width: 720px;
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
        margin: 0 0 var(--space-2) 0;
        text-align: center;
        font-family: var(--font-sans);
    }

    .runs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
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
