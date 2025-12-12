<script lang="ts">
    import type { CorrelationJobStatus as CorrelationJobStatusType, HarvestParams } from "../lib/api";

    interface Props {
        status: CorrelationJobStatusType | null;
        onSubmit: (params: HarvestParams) => void;
        submitting: boolean;
    }

    let { status, onSubmit, submitting }: Props = $props();

    let showParams = $state(false);
    let showConfig = $state(false);

    // Editable params with defaults
    let nBatches = $state(100);
    let batchSize = $state(256);
    let contextLength = $state(512);
    let ciThreshold = $state(1e-6);

    function formatTokenCount(n: number): string {
        if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
        if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
        return String(n);
    }

    function handleSubmit() {
        onSubmit({
            n_batches: nBatches,
            batch_size: batchSize,
            context_length: contextLength,
            ci_threshold: ciThreshold,
        });
        showConfig = false;
    }

    function formatNumber(n: number): string {
        if (n < 0.001) return n.toExponential(0);
        return String(n);
    }
</script>

<div class="correlation-section">
    {#if status === null}
        <span class="status-text muted">Correlations: Not computed</span>
        <div
            class="harvest-controls"
            role="group"
            onmouseenter={() => (showConfig = true)}
            onmouseleave={() => (showConfig = false)}
        >
            <button class="config-toggle" title="Configure harvest parameters">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path
                        d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
                    ></path>
                </svg>
            </button>
            <button class="harvest-button" onclick={handleSubmit} disabled={submitting}>
                {submitting ? "..." : "Harvest"}
            </button>

            {#if showConfig}
                <div class="config-dropdown">
                    <div class="config-content">
                        <div class="config-row">
                            <label for="n-batches">n_batches</label>
                            <input id="n-batches" type="number" bind:value={nBatches} min="1" max="10000" />
                        </div>
                        <div class="config-row">
                            <label for="batch-size">batch_size</label>
                            <input id="batch-size" type="number" bind:value={batchSize} min="1" max="1024" />
                        </div>
                        <div class="config-row">
                            <label for="context-length">context_length</label>
                            <input id="context-length" type="number" bind:value={contextLength} min="1" max="4096" />
                        </div>
                        <div class="config-row">
                            <label for="ci-threshold">ci_threshold</label>
                            <input
                                id="ci-threshold"
                                type="number"
                                bind:value={ciThreshold}
                                min="0"
                                max="1"
                                step="0.000001"
                            />
                        </div>
                        <div class="config-summary">
                            Total tokens: ~{formatTokenCount(nBatches * batchSize * contextLength)}
                        </div>
                    </div>
                </div>
            {/if}
        </div>
    {:else}
        <div
            class="correlation-status"
            role="group"
            onmouseenter={() => (showParams = true)}
            onmouseleave={() => (showParams = false)}
        >
            {#if status.status === "pending"}
                <span class="status-text pending">Correlations: Pending (job {status.job_id})</span>
            {:else if status.status === "running"}
                <span class="status-text running">Correlations: Running (job {status.job_id})</span>
            {:else if status.status === "completed"}
                <span class="status-text completed">
                    Correlations: Ready ({formatTokenCount(status.n_tokens)} tokens)
                </span>
            {:else if status.status === "failed"}
                <span class="status-text failed" title={status.error}> Correlations: Failed </span>
            {/if}

            {#if showParams}
                <div class="params-dropdown">
                    <div class="params-content">
                        <div class="param-row"><span>n_batches:</span> {status.params.n_batches}</div>
                        <div class="param-row"><span>batch_size:</span> {status.params.batch_size}</div>
                        <div class="param-row"><span>context_length:</span> {status.params.context_length}</div>
                        <div class="param-row">
                            <span>ci_threshold:</span>
                            {formatNumber(status.params.ci_threshold)}
                        </div>
                        {#if status.status === "completed"}
                            <div class="param-row"><span>components:</span> {status.n_components}</div>
                        {/if}
                        {#if (status.status === "pending" || status.status === "running") && status.last_log_line}
                            <div class="param-row log-line"><span>log:</span> {status.last_log_line}</div>
                        {/if}
                    </div>
                </div>
            {/if}
        </div>

        {#if status.status === "failed" || status.status === "completed"}
            <div
                class="harvest-controls"
                role="group"
                onmouseenter={() => (showConfig = true)}
                onmouseleave={() => (showConfig = false)}
            >
                <button class="config-toggle" title="Configure harvest parameters">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"></circle>
                        <!-- this is a gear, completely generated by claude 4.5 opus lol -->
                        <path
                            d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
                        ></path>
                    </svg>
                </button>
                <button class="harvest-button" onclick={handleSubmit} disabled={submitting}>
                    {submitting ? "..." : status.status === "completed" ? "Re-harvest" : "Harvest"}
                </button>

                {#if showConfig}
                    <div class="config-dropdown">
                        <div class="config-content">
                            <div class="config-row">
                                <label for="n-batches-retry">n_batches</label>
                                <input id="n-batches-retry" type="number" bind:value={nBatches} min="1" max="10000" />
                            </div>
                            <div class="config-row">
                                <label for="batch-size-retry">batch_size</label>
                                <input id="batch-size-retry" type="number" bind:value={batchSize} min="1" max="1024" />
                            </div>
                            <div class="config-row">
                                <label for="context-length-retry">context_length</label>
                                <input
                                    id="context-length-retry"
                                    type="number"
                                    bind:value={contextLength}
                                    min="1"
                                    max="4096"
                                />
                            </div>
                            <div class="config-row">
                                <label for="ci-threshold-retry">ci_threshold</label>
                                <input
                                    id="ci-threshold-retry"
                                    type="number"
                                    bind:value={ciThreshold}
                                    min="0"
                                    max="1"
                                    step="0.000001"
                                />
                            </div>
                            <div class="config-summary">
                                Total tokens: ~{formatTokenCount(nBatches * batchSize * contextLength)}
                            </div>
                        </div>
                    </div>
                {/if}
            </div>
        {/if}
    {/if}
</div>

<style>
    .correlation-section {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding-left: var(--space-2);
        position: relative;
    }

    .correlation-status {
        position: relative;
    }

    .status-text {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        white-space: nowrap;
        cursor: default;
    }

    .status-text.muted {
        color: var(--text-muted);
    }

    .status-text.pending,
    .status-text.running {
        color: var(--accent-primary);
    }

    .status-text.completed {
        color: var(--status-positive);
    }

    .status-text.failed {
        color: var(--status-negative-bright);
    }

    .params-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        padding-top: var(--space-2);
        z-index: 1000;
    }

    .params-content {
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: var(--space-2) var(--space-3);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-primary);
        min-width: 180px;
    }

    .param-row {
        display: flex;
        justify-content: space-between;
        gap: var(--space-3);
        padding: var(--space-1) 0;
    }

    .param-row span {
        color: var(--text-muted);
    }

    .harvest-controls {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        position: relative;
    }

    .config-toggle {
        padding: var(--space-1);
        color: var(--text-muted);
        background: transparent;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .config-toggle:hover {
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .harvest-button {
        padding: var(--space-1) var(--space-3);
        color: var(--text-primary);
        border: 1px solid var(--border-default);
        font-weight: 500;
        white-space: nowrap;
    }

    .harvest-button:hover:not(:disabled) {
        background: var(--accent-primary);
        color: white;
    }

    .harvest-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
    }

    .config-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: var(--space-2);
        z-index: 1000;
    }

    .config-content {
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: var(--space-3);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-primary);
        min-width: 220px;
    }

    .config-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-1) 0;
    }

    .config-row label {
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .config-row input {
        width: 90px;
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-primary);
        color: var(--text-primary);
        text-align: right;
    }

    .config-row input:focus {
        outline: none;
        border-color: var(--accent-primary);
    }

    .config-summary {
        margin-top: var(--space-2);
        padding-top: var(--space-2);
        border-top: 1px solid var(--border-default);
        color: var(--text-muted);
        font-size: var(--text-xs);
    }
</style>
