<script lang="ts">
    import type { LoadedRun } from "../lib/api";
    import { loadClusterMapping } from "../lib/api";
    import { clusterMapping } from "../lib/clusterMapping.svelte";

    type Props = {
        loadedRun: LoadedRun | null;
    };

    let { loadedRun }: Props = $props();

    let inputPath = $state("");
    let loading = $state(false);
    let error = $state<string | null>(null);

    // Clear cluster mapping when run changes
    $effect(() => {
        clusterMapping.clearIfRunChanged(loadedRun?.wandb_path ?? null);
    });

    async function handleLoad() {
        const path = inputPath.trim();
        if (!path || !loadedRun) return;

        loading = true;
        error = null;
        try {
            const result = await loadClusterMapping(path);
            clusterMapping.setMapping(result.mapping, path, loadedRun.wandb_path);
            inputPath = "";
        } catch (e) {
            error = e instanceof Error ? e.message : "Failed to load";
        } finally {
            loading = false;
        }
    }

    function handleClear() {
        clusterMapping.clear();
        error = null;
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === "Enter") {
            handleLoad();
        }
    }
</script>

{#if loadedRun}
    <div class="cluster-input-wrapper">
        {#if clusterMapping.filePath}
            <div class="cluster-loaded">
                <span class="cluster-label">Clusters:</span>
                <span class="cluster-path" title={clusterMapping.filePath}>
                    {clusterMapping.filePath.split("/").pop()}
                </span>
                <button type="button" class="clear-button" onclick={handleClear} title="Clear cluster mapping">
                    x
                </button>
            </div>
        {:else}
            <div class="cluster-form">
                <input
                    type="text"
                    placeholder="path/to/cluster_mapping_<id>.json"
                    bind:value={inputPath}
                    onkeydown={handleKeydown}
                    disabled={loading}
                />
                <button type="button" onclick={handleLoad} disabled={loading || !inputPath.trim()}>
                    {loading ? "..." : "Load"}
                </button>
            </div>
            {#if error}
                <span class="error-text" title={error}>Error</span>
            {/if}
        {/if}
    </div>
{/if}

<style>
    .cluster-input-wrapper {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .cluster-form {
        display: flex;
        align-items: center;
        gap: var(--space-1);
    }

    .cluster-form input {
        width: 260px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .cluster-form input::placeholder {
        color: var(--text-muted);
    }

    .cluster-form input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .cluster-form button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-size: var(--text-xs);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .cluster-form button:hover:not(:disabled) {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .cluster-form button:disabled {
        cursor: not-allowed;
        opacity: 0.5;
    }

    .cluster-loaded {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--accent-primary-dim);
        border-radius: var(--radius-sm);
    }

    .cluster-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .cluster-path {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        max-width: 150px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .clear-button {
        padding: 0 var(--space-1);
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
        line-height: 1;
    }

    .clear-button:hover {
        color: var(--text-primary);
    }

    .error-text {
        font-size: var(--text-xs);
        color: var(--accent-negative);
        cursor: help;
    }
</style>
