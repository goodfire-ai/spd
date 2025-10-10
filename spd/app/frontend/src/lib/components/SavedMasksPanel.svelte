<script lang="ts">
    import { onMount } from "svelte";
    import * as api from "$lib/api";
    import type { MaskOverrideDTO } from "$lib/api";

    export let onApplyMask: (maskId: string) => void;

    let savedMasks: MaskOverrideDTO[] = [];
    let loading = false;

    export async function loadMasks() {
        loading = true;
        try {
            savedMasks = await api.getMaskOverrides();
        } catch (error) {
            console.error("Failed to load mask overrides:", error);
        } finally {
            loading = false;
        }
    }

    function handleApply(maskId: string) {
        onApplyMask(maskId);
    }

    onMount(() => {
        loadMasks();
    });
</script>

<div class="saved-masks-panel">
    <div class="panel-header">
        <h3>Saved Masks</h3>
        <button class="refresh-btn" on:click={loadMasks} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
        </button>
    </div>

    {#if savedMasks.length === 0}
        <div class="empty-state">
            No saved masks yet. Create masks using multi-select mode below.
        </div>
    {:else}
        <div class="masks-container">
            {#each savedMasks as mask}
                <div class="mask-card">
                    <div class="mask-info">
                        <div class="mask-description">
                            {mask.description || "Unnamed mask"}
                        </div>
                        <div class="mask-details">
                            <span class="mask-layer">Layer: {mask.layer}</span>
                            <span class="mask-l0">L0: {mask.combined_mask.l0}</span>
                        </div>
                    </div>
                    <button class="apply-btn" on:click={() => handleApply(mask.id)}>
                        Apply as Ablation
                    </button>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .saved-masks-panel {
        margin-bottom: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }

    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .panel-header h3 {
        margin: 0;
        font-size: 1.1rem;
        color: #333;
    }

    .refresh-btn {
        padding: 0.25rem 0.75rem;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
    }

    .refresh-btn:hover:not(:disabled) {
        background: #0056b3;
    }

    .refresh-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .empty-state {
        padding: 2rem;
        text-align: center;
        color: #6c757d;
        font-style: italic;
    }

    .masks-container {
        display: flex;
        gap: 0.75rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }

    .mask-card {
        flex: 0 0 auto;
        min-width: 200px;
        padding: 0.75rem;
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 6px;
        transition: all 0.2s;
    }

    .mask-card:hover {
        border-color: #007bff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .mask-info {
        margin-bottom: 0.75rem;
    }

    .mask-description {
        font-weight: 500;
        color: #333;
        margin-bottom: 0.5rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .mask-details {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        font-size: 0.85rem;
        color: #6c757d;
    }

    .mask-layer {
        font-weight: 500;
    }

    .mask-l0 {
        color: #007bff;
    }

    .apply-btn {
        width: 100%;
        padding: 0.5rem;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: 500;
        transition: background 0.2s;
    }

    .apply-btn:hover {
        background: #0056b3;
    }

    /* Scrollbar styling */
    .masks-container::-webkit-scrollbar {
        height: 6px;
    }

    .masks-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }

    .masks-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }

    .masks-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
