<script lang="ts">
    import type { Loadable } from "../../lib";
    import type { InterpretationDetail } from "../../lib/api";
    import type { InterpretationBackendState } from "../../lib/useRun.svelte";

    interface Props {
        interpretation: Loadable<InterpretationBackendState>;
        interpretationDetail: Loadable<InterpretationDetail | null>;
        onGenerate: () => void;
    }

    let { interpretation, interpretationDetail, onGenerate }: Props = $props();

    let showPrompt = $state(false);
</script>

<div class="interpretation-container">
    <div
        class="interpretation-badge"
        class:loading={(interpretation.status === "loaded" && interpretation.data.status === "generating") ||
            interpretation.status === "loading"}
    >
        {#if interpretation.status === "loading"}
            <span class="interpretation-label loading-text">Loading interpretations...</span>
        {:else if interpretation.status === "loaded"}
            {@const interpretationData = interpretation.data}
            {#if interpretationData.status === "none"}
                <button class="generate-btn" onclick={onGenerate}>Generate Interpretation</button>
            {:else if interpretationData.status === "generating"}
                <span class="interpretation-label loading-text">Generating interpretation...</span>
            {:else if interpretationData.status === "generated"}
                <div class="interpretation-content">
                    <div class="interpretation-header">
                        <span class="interpretation-label">{interpretationData.data.label}</span>
                        <span class="confidence confidence-{interpretationData.data.confidence}"
                            >{interpretationData.data.confidence}</span
                        >
                    </div>
                    {#if interpretationDetail.status === "loaded" && interpretationDetail.data?.reasoning}
                        <span class="interpretation-reasoning">{interpretationDetail.data.reasoning}</span>
                    {:else if interpretationDetail.status === "loading"}
                        <span class="interpretation-reasoning loading-text">Loading reasoning...</span>
                    {/if}
                </div>
                <button class="prompt-toggle" onclick={() => (showPrompt = !showPrompt)}>
                    {showPrompt ? "Hide" : "View"} Autointerp Prompt
                </button>
                <!-- Error state for generating -->
            {:else if interpretationData.status === "generation-error"}
                <span class="interpretation-label error-text">{String(interpretationData.error)}</span>
                <button class="retry-btn" onclick={onGenerate}>Retry</button>
            {/if}
            <!-- Error state for fetching -->
        {:else if interpretation.status === "error"}
            <span class="interpretation-label error-text">{String(interpretation.error)}</span>
        {/if}
    </div>

    {#if showPrompt}
        <div class="prompt-display">
            {#if interpretationDetail.status === "loading"}
                <span class="loading-text">Loading prompt...</span>
            {:else if interpretationDetail.status === "error"}
                <span class="error-text">Error loading prompt: {String(interpretationDetail.error)}</span>
            {:else if interpretationDetail.status === "loaded" && interpretationDetail.data}
                <pre>{interpretationDetail.data.prompt}</pre>
            {:else}
                <span class="loading-text">Loading prompt...</span>
            {/if}
        </div>
    {/if}
</div>

<style>
    .interpretation-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .interpretation-badge {
        display: flex;
        align-items: flex-start;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--color-accent, #6366f1);
    }

    .interpretation-content {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        flex: 1;
        min-width: 0;
    }

    .interpretation-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .interpretation-reasoning {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        line-height: 1.4;
    }

    .interpretation-badge.loading {
        opacity: 0.7;
        border-left-color: var(--text-muted);
    }

    .interpretation-label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: var(--text-sm);
    }

    .interpretation-label.loading-text {
        color: var(--text-muted);
        font-style: italic;
    }

    .interpretation-label.error-text {
        color: var(--status-negative);
    }

    .interpretation-label.muted {
        color: var(--text-muted);
        font-style: italic;
    }

    .confidence {
        font-size: var(--text-xs);
        padding: 2px 6px;
        border-radius: var(--radius-sm);
        text-transform: uppercase;
        font-weight: 600;
    }

    .confidence-high {
        background: color-mix(in srgb, #22c55e 20%, transparent);
        color: #22c55e;
    }

    .confidence-medium {
        background: color-mix(in srgb, #eab308 20%, transparent);
        color: #eab308;
    }

    .confidence-low {
        background: color-mix(in srgb, var(--text-muted) 20%, transparent);
        color: var(--text-muted);
    }

    .generate-btn,
    .retry-btn {
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-weight: 500;
    }

    .generate-btn:hover,
    .retry-btn:hover {
        background: var(--accent-primary-dim);
    }

    .retry-btn {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }

    .retry-btn:hover {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .prompt-toggle {
        margin-left: auto;
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-weight: 500;
    }

    .prompt-toggle:hover {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .prompt-display {
        background: var(--bg-primary);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: var(--space-3);
        max-height: 400px;
        overflow: auto;
    }

    .prompt-display pre {
        margin: 0;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--text-secondary);
    }
</style>
