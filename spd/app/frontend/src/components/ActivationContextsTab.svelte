<script lang="ts">
    import type { Loadable } from "../lib";
    import * as api from "../lib/api";
    import type { ActivationContextsSummary } from "../lib/localAttributionsTypes";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";
    import StatusText from "./ui/StatusText.svelte";

    let summary = $state<Loadable<ActivationContextsSummary>>(null);

    $effect(() => {
        summary = { status: "loading" };
        api.getActivationContextsSummary()
            .then((data) => {
                summary = { status: "loaded", data };
            })
            .catch((error) => {
                summary = { status: "error", error };
            });
    });
</script>

<div class="tab-wrapper">
    {#if summary === null || summary.status === "loading"}
        <div class="loading">Loading activation contexts summary...</div>
    {:else if summary.status === "error"}
        <StatusText>Error loading summary: {String(summary.error)}</StatusText>
    {:else}
        <ActivationContextsViewer activationContextsSummary={summary.data} />
    {/if}
</div>

<style>
    .tab-wrapper {
        height: 100%;
    }

    .loading {
        padding: var(--space-4);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
