<script lang="ts">
    import type { ActivationContextsSummary } from "../lib/localAttributionsTypes";
    // import ActivationContextsViewer from "./ActivationContextsViewer.svelte";

    interface Props {
        activationContextsSummary: ActivationContextsSummary | null;
    }

    let { activationContextsSummary }: Props = $props();
</script>

<div class="tab-content">
    {#if activationContextsSummary === null}
        <div class="empty-state">
            <p>No activation contexts available.</p>
            <p class="hint">Run the harvest pipeline first:</p>
            <code>python -m spd.autointerp.scripts.run_autointerp harvest &lt;wandb_path&gt;</code>
        </div>
    {:else}
        broken 😭

        <!-- <ActivationContextsViewer harvestMetadata={{ layers: activationContextsSummary }} /> -->
    {/if}
</div>

<style>
    .tab-content {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        padding: var(--space-6);
    }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        gap: var(--space-2);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .empty-state p {
        margin: 0;
    }

    .hint {
        font-size: var(--text-sm);
    }

    .empty-state code {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        background: var(--bg-elevated);
        padding: var(--space-2) var(--space-3);
        border-radius: var(--radius-md);
        color: var(--text-primary);
    }
</style>
