<script lang="ts">
    import type { CorrelationJobStatus } from "../lib/api";
    import * as api from "../lib/api";
    import { onMount } from "svelte";

    let status = $state<CorrelationJobStatus | null>(null);
    let submitting = $state(false);
    let showParams = $state(false);

    export async function reload() {
        try {
            status = await api.getCorrelationJobStatus();
        } catch (e) {
            console.error("Failed to load correlation job status:", e);
            status = null;
        }
    }

    onMount(reload);

    // Poll while pending/running
    $effect(() => {
        const s = status?.status;
        if (s !== "pending" && s !== "running") return;

        const interval = setInterval(reload, 2000);
        return () => clearInterval(interval);
    });

    async function submit() {
        if (submitting) return;
        submitting = true;
        try {
            await api.submitCorrelationJob();
            await reload();
        } catch (e) {
            console.error("Failed to submit correlation job:", e);
        } finally {
            submitting = false;
        }
    }

    function formatTokenCount(n: number): string {
        if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
        if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
        return String(n);
    }
</script>

<div class="correlation-section">
    {#if status === null}
        <span class="status-text muted">Correlations: Not computed</span>
        <button class="harvest-button" onclick={submit} disabled={submitting}>
            {submitting ? "..." : "Harvest"}
        </button>
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
                <span class="status-text failed" title={status.error}>
                    Correlations: Failed
                </span>
            {/if}

            {#if showParams}
                <div class="params-dropdown">
                    <div class="params-content">
                        <div class="param-row"><span>n_batches:</span> {status.params.n_batches}</div>
                        <div class="param-row"><span>batch_size:</span> {status.params.batch_size}</div>
                        <div class="param-row"><span>context_length:</span> {status.params.context_length}</div>
                        <div class="param-row"><span>ci_threshold:</span> {status.params.ci_threshold}</div>
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

        {#if status.status === "failed"}
            <button class="harvest-button" onclick={submit} disabled={submitting}>
                {submitting ? "..." : "Harvest"}
            </button>
        {/if}
    {/if}
</div>

<style>
    .correlation-section {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin-left: var(--space-2);
        padding-left: var(--space-2);
        border-left: 1px solid var(--border-default);
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

    .harvest-button {
        padding: var(--space-1) var(--space-2);
        background: var(--accent-primary);
        color: white;
        border: 1px solid var(--accent-primary);
        border-radius: var(--radius-sm);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        font-weight: 500;
        cursor: pointer;
        white-space: nowrap;
    }

    .harvest-button:hover:not(:disabled) {
        background: var(--accent-primary-bright);
        border-color: var(--accent-primary-bright);
    }

    .harvest-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
</style>
