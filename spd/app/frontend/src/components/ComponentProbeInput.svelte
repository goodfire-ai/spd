<script lang="ts">
    import type { ComponentProbeResult } from "../lib/localAttributionsTypes";
    import { getTokenHighlightBg } from "../lib/colors";
    import { probeComponent } from "../lib/localAttributionsApi";

    interface Props {
        layer: string;
        componentIdx: number;
    }

    let { layer, componentIdx }: Props = $props();

    let probeText = $state("");
    let probeResult = $state<ComponentProbeResult | null>(null);
    let probeLoading = $state(false);
    let probeError = $state<string | null>(null);
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;

    async function runProbe(text: string) {
        if (!text.trim()) {
            probeResult = null;
            probeError = null;
            return;
        }

        probeLoading = true;
        probeError = null;

        try {
            probeResult = await probeComponent(text, layer, componentIdx);
        } catch (e) {
            probeError = e instanceof Error ? e.message : "Failed to probe component";
            probeResult = null;
        } finally {
            probeLoading = false;
        }
    }

    function onProbeInput(e: Event) {
        const target = e.target as HTMLInputElement;
        probeText = target.value;

        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => runProbe(probeText), 100);
    }

    // Re-run probe when layer or component changes (if there's text)
    $effect(() => {
        layer; // eslint-disable-line @typescript-eslint/no-unused-expressions
        componentIdx; // eslint-disable-line @typescript-eslint/no-unused-expressions
        probeResult = null;
        probeError = null;
        if (probeText.trim()) {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => runProbe(probeText), 100);
        }
    });
</script>

<div class="probe-section">
    <h5>Test Custom Text</h5>
    <input
        type="text"
        class="probe-input"
        placeholder="Enter text to test..."
        value={probeText}
        oninput={onProbeInput}
    />
    {#if probeLoading}
        <p class="probe-status">Loading...</p>
    {:else if probeError}
        <p class="probe-error">{probeError}</p>
    {:else if probeResult && probeResult.tokens.length > 0}
        <div class="probe-result">
            <span class="probe-tokens"
                >{#each probeResult.tokens as tok, i (i)}<span
                        class="probe-token"
                        style="background-color:{getTokenHighlightBg(probeResult.ci_values[i])}"
                        title="CI: {probeResult.ci_values[i].toFixed(4)}">{tok}</span
                    >{/each}</span
            >
        </div>
    {/if}
</div>

<style>
    .probe-section {
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
    }

    h5 {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .probe-input {
        width: 100%;
        padding: var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        color: var(--text-primary);
        box-sizing: border-box;
    }

    .probe-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .probe-input::placeholder {
        color: var(--text-muted);
    }

    .probe-status {
        margin: var(--space-2) 0 0 0;
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .probe-error {
        margin: var(--space-2) 0 0 0;
        font-size: var(--text-sm);
        color: var(--status-negative);
        font-family: var(--font-mono);
    }

    .probe-result {
        margin-top: var(--space-2);
        overflow-x: auto;
    }

    .probe-tokens {
        display: inline;
        white-space: pre-wrap;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
    }

    .probe-token {
        display: inline;
        padding: 1px 0;
        margin-right: 1px;
        border-right: 1px solid var(--border-subtle);
        white-space: pre;
        cursor: help;
    }
</style>
