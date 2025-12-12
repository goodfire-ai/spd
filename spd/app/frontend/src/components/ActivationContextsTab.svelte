<script lang="ts">
    import * as api from "../lib/api";
    import * as attrApi from "../lib/localAttributionsApi";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";

    type Props = {
        onHarvestComplete: () => void;
    };

    let { onHarvestComplete }: Props = $props();

    let loading = $state(false);
    let harvestMetadata = $state<api.HarvestMetadata | null>(null);
    let progress = $state<api.ProgressUpdate | null>(null);

    // Configuration - single object, prefilled from cache if available
    let config = $state<attrApi.ActivationContextsConfig>({
        n_batches: 10,
        batch_size: 16,
        n_tokens_either_side: 8,
        separation_tokens: 8,
        importance_threshold: 0.0,
        topk_examples: 200,
    });

    // Load cached activation contexts and config on mount
    $effect(() => {
        loadCachedData();
    });

    async function loadCachedData() {
        // Load config first - if it exists, contexts exist too
        const cachedConfig = await attrApi.getActivationContextsConfig();
        if (cachedConfig === null) return;

        config = cachedConfig;
        const summary = await attrApi.getActivationContextsSummary();
        harvestMetadata = { layers: summary };
    }

    async function loadContexts() {
        loading = true;
        harvestMetadata = null;
        progress = null;
        try {
            const data = await api.getSubcomponentActivationContexts(config, (p) => {
                progress = p;
            });
            harvestMetadata = data;
            onHarvestComplete();
        } finally {
            loading = false;
            progress = null;
        }
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && !loading) {
            loadContexts();
        }
    }
</script>

<div class="tab-content">
    <div class="config-box">
        <div class="config-header">
            <span class="config-title">Configuration</span>
            <button class="harvest-button" onclick={loadContexts} disabled={loading}>
                {loading ? "Harvesting..." : "Harvest"}
            </button>
        </div>
        <div class="config-grid">
            <div class="config-item">
                <label for="n-steps">Batches</label>
                <input
                    id="n-steps"
                    type="number"
                    step="1"
                    min="1"
                    bind:value={config.n_batches}
                    onkeydown={handleKeydown}
                />
            </div>
            <div class="config-item">
                <label for="batch-size">Batch Size</label>
                <input
                    id="batch-size"
                    type="number"
                    step="1"
                    min="1"
                    bind:value={config.batch_size}
                    onkeydown={handleKeydown}
                />
            </div>
            <div class="config-item">
                <label for="n-tokens">Context Window</label>
                <input
                    id="n-tokens"
                    type="number"
                    step="1"
                    min="0"
                    bind:value={config.n_tokens_either_side}
                    onkeydown={handleKeydown}
                />
            </div>
            <div class="config-item">
                <label for="separation-tokens">Separation</label>
                <input
                    id="separation-tokens"
                    type="number"
                    step="1"
                    min="0"
                    bind:value={config.separation_tokens}
                    onkeydown={handleKeydown}
                />
            </div>
            <div class="config-item">
                <label for="importance-threshold">CI Threshold</label>
                <input
                    id="importance-threshold"
                    type="number"
                    step="0.001"
                    min="0"
                    max="1"
                    bind:value={config.importance_threshold}
                    onkeydown={handleKeydown}
                />
            </div>
            <div class="config-item">
                <label for="topk-examples">Top K</label>
                <input
                    id="topk-examples"
                    type="number"
                    step="1"
                    min="1"
                    bind:value={config.topk_examples}
                    onkeydown={handleKeydown}
                />
            </div>
        </div>
        {#if loading && progress}
            <div class="progress-section">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress.progress * 100}%"></div>
                </div>
                <span class="progress-text">{(progress.progress * 100).toFixed(0)}%</span>
            </div>
        {/if}
    </div>

    <div class="results-box">
        {#if loading && !progress}
            <div class="empty-state">Loading...</div>
        {:else if harvestMetadata}
            <ActivationContextsViewer {harvestMetadata} />
        {:else}
            <div class="empty-state">
                <p>No activation contexts loaded</p>
                <p class="hint">Click Harvest to generate contexts from training data</p>
            </div>
        {/if}
    </div>
</div>

<style>
    .tab-content {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        gap: var(--space-4);
        padding: var(--space-6);
    }

    .config-box {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
    }

    .config-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .config-title {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .config-grid {
        display: flex;
        gap: var(--space-4);
        flex-wrap: wrap;
    }

    .config-item {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .config-item label {
        font-size: var(--text-xs);
        font-family: var(--font-sans);
        color: var(--text-muted);
        font-weight: 500;
    }

    .config-item input {
        width: 80px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
    }

    .config-item input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .harvest-button {
        padding: var(--space-1) var(--space-3);
        border: none;
        background: var(--accent-primary);
        color: white;
        font-weight: 500;
        font-size: var(--text-sm);
    }

    .harvest-button:hover:not(:disabled) {
        background: var(--accent-primary-dim);
    }

    .harvest-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
    }

    .progress-section {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .progress-bar {
        flex: 1;
        height: 4px;
        background: var(--border-default);
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: var(--accent-primary);
        transition: width 0.15s ease-out;
    }

    .progress-text {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        min-width: 3ch;
    }

    .results-box {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
        overflow-y: auto;
    }

    .empty-state {
        display: flex;
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        text-align: center;
        font-family: var(--font-sans);
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
    }

    .empty-state .hint {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
</style>
