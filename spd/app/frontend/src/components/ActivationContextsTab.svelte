<script lang="ts">
    import { getContext } from "svelte";
    import type { Loadable } from "../lib";
    import type { ActivationContextsSummary } from "../lib/localAttributionsTypes";
    import { RUN_STATE_KEY, type RunStateContext } from "../lib/runState.svelte";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";
    import StatusText from "./ui/StatusText.svelte";

    const runState = getContext<RunStateContext>(RUN_STATE_KEY);

    let summary = $state<Loadable<ActivationContextsSummary>>({ status: "uninitialized" });

    $effect(() => {
        // Re-fetch when run changes
        if (runState.run?.status !== "loaded") return;

        summary = { status: "loading" };
        runState
            .getActivationContextsSummary()
            .then((data) => {
                summary = { status: "loaded", data };
            })
            .catch((error) => {
                summary = { status: "error", error };
            });
    });
</script>

<div class="tab-wrapper">
    {#if summary.status === "uninitialized" || summary.status === "loading"}
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
        padding: var(--space-4);
    }

    .loading {
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
