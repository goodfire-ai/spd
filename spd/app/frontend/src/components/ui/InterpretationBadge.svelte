<script lang="ts">
    import type { Loadable } from "../../lib";
    import { requestComponentInterpretation, type Interpretation } from "../../lib/api";

    interface Props {
        interpretation: Loadable<Interpretation>;
        layer?: string;
        cIdx?: number;
        onInterpretationGenerated?: (interp: Interpretation) => void;
    }

    let { interpretation, layer, cIdx, onInterpretationGenerated }: Props = $props();

    let requesting = $state(false);
    let requestError = $state<string | null>(null);
    let showPrompt = $state(false);

    const canRequest = $derived(layer !== undefined && cIdx !== undefined && onInterpretationGenerated !== undefined);

    async function handleRequestInterpretation() {
        if (!canRequest || requesting) return;
        requesting = true;
        requestError = null;

        try {
            const result = await requestComponentInterpretation(layer!, cIdx!);
            onInterpretationGenerated!(result);
        } catch (e) {
            requestError = e instanceof Error ? e.message : String(e);
        } finally {
            requesting = false;
        }
    }
</script>

<div class="interpretation-container">
    <div class="interpretation-badge" class:loading={requesting}>
        {#if interpretation?.status === "loaded"}
            <span class="interpretation-label">{interpretation.data.label}</span>
            <span class="confidence confidence-{interpretation.data.confidence}">{interpretation.data.confidence}</span>
            <button class="prompt-toggle" onclick={() => (showPrompt = !showPrompt)}>
                {showPrompt ? "Hide" : "View"} Prompt
            </button>
        {:else if interpretation?.status === "loading" || requesting}
            <span class="interpretation-label loading-text">Generating interpretation...</span>
        {:else if interpretation?.status === "error" || requestError}
            <span class="interpretation-label error-text">{requestError || String(interpretation?.error)}</span>
            {#if canRequest}
                <button class="retry-btn" onclick={handleRequestInterpretation}>Retry</button>
            {/if}
        {:else if interpretation === null && canRequest}
            <button class="generate-btn" onclick={handleRequestInterpretation}>Generate Interpretation</button>
        {:else if interpretation === null}
            <span class="interpretation-label muted">No interpretation available</span>
        {:else}
            <span class="interpretation-label muted">Something went wrong</span>
        {/if}
    </div>

    {#if showPrompt && interpretation?.status === "loaded"}
        <div class="prompt-display">
            <pre>{interpretation.data.prompt}</pre>
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
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--color-accent, #6366f1);
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
