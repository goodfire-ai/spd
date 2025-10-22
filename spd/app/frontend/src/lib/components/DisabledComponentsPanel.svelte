<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import { ablationComponentMask } from "$lib/stores/componentState";

    export let promptTokens: string[];
    export let isLoading: boolean;
    export let onSendAblation: () => void;
    export let onToggleComponent: (layerName: string, tokenIdx: number, componentIdx: number) => void;

    $: hasDisabledComponents = Object.keys($ablationComponentMask).some((layer) =>
        $ablationComponentMask[layer].some((tokenList) => tokenList.length > 0)
    );
</script>

<div class="disabled-components-panel">
    <div class="disabled-header-row">
        <h3>Disabled Components</h3>
        <button
            on:click={onSendAblation}
            disabled={isLoading}
            class="ablate-button"
        >
            {isLoading ? "Sending..." : "Run with ablations"}
        </button>
    </div>
    {#if hasDisabledComponents}
        <div class="disabled-list">
            {#each Object.entries($ablationComponentMask) as [layerName, tokenArrays]}
                {#each tokenArrays as disabledComponents, tokenIdx}
                    {#if disabledComponents.length > 0}
                        <div class="disabled-group">
                            <div class="disabled-header">
                                <strong>{promptTokens[tokenIdx]}</strong>
                                in
                                <em>{layerName}</em>
                            </div>
                            <div class="disabled-items">
                                {#each disabledComponents as componentIdx}
                                    <span
                                        class="disabled-chip"
                                        on:click={() =>
                                            onToggleComponent(
                                                layerName,
                                                tokenIdx,
                                                componentIdx
                                            )}
                                    >
                                        {componentIdx} ×
                                    </span>
                                {/each}
                            </div>
                        </div>
                    {/if}
                {/each}
            {/each}
        </div>
    {:else}
        <p class="no-disabled">No components disabled yet</p>
    {/if}
</div>

<style>
    .disabled-components-panel {
        flex: 0 0 300px;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: #f8f9fa;
        margin-top: 1rem;
        /* max-height: 500px; */
        /* overflow-y: auto; */
    }

    .disabled-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .disabled-components-panel h3 {
        margin: 0;
        color: #333;
        font-size: 1.1rem;
    }

    .ablate-button {
        background-color: #ff6b35;
        padding: 0.5rem 1rem;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .ablate-button:hover:not(:disabled) {
        background-color: #e55a2b;
    }

    .ablate-button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }

    .disabled-list {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .disabled-group {
        background: white;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }

    .disabled-header {
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #555;
    }

    .disabled-items {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
    }

    .disabled-chip {
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 12px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }

    .disabled-chip:hover {
        background: #ff5252;
    }

    .no-disabled {
        color: #999;
        font-style: italic;
        margin: 0;
        text-align: center;
    }
</style>