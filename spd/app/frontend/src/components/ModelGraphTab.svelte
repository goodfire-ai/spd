<script lang="ts">
    import { onMount } from "svelte";
    import type { Loadable } from "../lib";
    import { getModelGraph, type ModelGraphResponse } from "../lib/api";
    import ModelGraph from "./ModelGraph.svelte";

    let data = $state<Loadable<ModelGraphResponse>>({ status: "uninitialized" });

    onMount(async () => {
        data = { status: "loading" };
        try {
            const result = await getModelGraph();
            data = { status: "loaded", data: result };
        } catch (e) {
            data = { status: "error", error: e };
        }
    });
</script>

<div class="model-graph-tab">
    {#if data.status === "loading"}
        <div class="status">Loading model graph...</div>
    {:else if data.status === "error"}
        <div class="status error">Failed to load graph: {String(data.error)}</div>
    {:else if data.status === "loaded"}
        <ModelGraph nodes={data.data.nodes} edges={data.data.edges} />
    {:else}
        <div class="status">Initializing...</div>
    {/if}
</div>

<style>
    .model-graph-tab {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .status {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: var(--text-muted);
        font-family: var(--font-sans);
        font-size: var(--text-sm);
    }

    .status.error {
        color: var(--status-negative);
    }
</style>
