<script lang="ts">
    import type { SubcomponentProbeResult } from "../lib/promptAttributionsTypes";
    import { probeComponent } from "../lib/api";
    import TokenHighlights from "./TokenHighlights.svelte";

    interface Props {
        layer: string;
        componentIdx: number;
        maxAbsComponentAct: number;
    }

    let { layer, componentIdx, maxAbsComponentAct }: Props = $props();

    let probeText = $state("");
    let probeResult = $state<SubcomponentProbeResult | null>(null);
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
        void [layer, componentIdx]; // track dependencies
        probeResult = null;
        probeError = null;
        if (probeText.trim()) {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => runProbe(probeText), 100);
        }
    });
</script>

<div class="probe-section">
    <input
        type="text"
        class="probe-input"
        placeholder="Enter a custom prompt here...."
        value={probeText}
        oninput={onProbeInput}
    />
    {#if probeLoading}
        <p class="probe-status">Loading...</p>
    {:else if probeError}
        <p class="probe-error">{probeError}</p>
    {:else if probeResult && probeResult.tokens.length > 0}
        <div class="probe-result">
            <TokenHighlights
                tokenStrings={probeResult.tokens}
                tokenCi={probeResult.ci_values}
                tokenComponentActs={probeResult.subcomp_acts}
                tokenNextProbs={probeResult.next_token_probs}
                {maxAbsComponentAct}
            />
        </div>
    {/if}
</div>

<style>
    .probe-section {
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
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
        padding: var(--space-2) var(--space-2) 30px var(--space-2);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        overflow-x: auto;
        font-size: var(--text-xs);
    }

    .probe-result :global(.token-highlight) {
        font-size: var(--text-xs);
    }
</style>
