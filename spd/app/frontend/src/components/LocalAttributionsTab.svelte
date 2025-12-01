<script lang="ts">
    import { API_URL } from "../lib/api";
    import LocalAttributionsViewer from "./LocalAttributionsViewer.svelte";
    import type { LocalAttributionsData } from "./localAttributionsTypes";

    let data = $state<LocalAttributionsData | null>(null);
    let loading = $state(false);
    let error = $state<string | null>(null);

    async function loadData() {
        loading = true;
        error = null;
        try {
            // The backend serves the JSON file directly in the component's expected format
            const response = await fetch(`${API_URL}/local_attributions`);
            if (!response.ok) {
                throw new Error(`Failed to load: ${response.statusText}`);
            }
            data = await response.json() as LocalAttributionsData;
        } catch (e) {
            error = e instanceof Error ? e.message : "Unknown error";
        } finally {
            loading = false;
        }
    }

    $effect(() => {
        loadData();
    });
</script>

<div class="local-attributions-tab">
    {#if loading}
        <div class="loading">Loading local attributions...</div>
    {:else if error}
        <div class="error">{error}</div>
    {:else if data}
        <LocalAttributionsViewer {data} />
    {:else}
        <div class="no-data">No data available</div>
    {/if}
</div>

<style>
    .local-attributions-tab {
        height: 100%;
        overflow: hidden;
    }

    .loading,
    .error,
    .no-data {
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem;
    }

    .loading {
        background: #e3f2fd;
        color: #1565c0;
    }

    .error {
        background: #ffebee;
        color: #c62828;
    }

    .no-data {
        background: #f5f5f5;
        color: #666;
    }
</style>
