<script lang="ts">
    import { multiSelectMode, selectedTokensForCombining, combinedMasks } from "$lib/stores/componentState";
    import { api } from "$lib/api";

    let combining = false;
    let description = "";
    let simulatedL0: number | null = null;
    let simulating = false;

    function toggleMultiSelectMode() {
        $multiSelectMode = !$multiSelectMode;
        if (!$multiSelectMode) {
            // Clear selections when exiting multi-select mode
            $selectedTokensForCombining = [];
            simulatedL0 = null;
        }
    }

    function clearSelections() {
        $selectedTokensForCombining = [];
        simulatedL0 = null;
    }

    // Reactively simulate merge whenever selections change
    $: if ($selectedTokensForCombining.length > 0 && getLayersWithSelections().length === 1) {
        simulateMergeL0();
    } else {
        simulatedL0 = null;
    }

    async function simulateMergeL0() {
        const layers = getLayersWithSelections();
        if (layers.length !== 1) {
            simulatedL0 = null;
            return;
        }

        const layer = layers[0];
        const tokenIndices = $selectedTokensForCombining
            .filter((t) => t.layer === layer)
            .map((t) => t.tokenIdx);

        if (tokenIndices.length === 0) {
            simulatedL0 = null;
            return;
        }

        simulating = true;
        try {
            const response = await api.simulateMerge({
                layer,
                token_indices: tokenIndices,
            });
            simulatedL0 = response.l0;
        } catch (error) {
            console.error("Failed to simulate merge:", error);
            simulatedL0 = null;
        } finally {
            simulating = false;
        }
    }

    function getLayersWithSelections(): string[] {
        const layers = new Set<string>();
        $selectedTokensForCombining.forEach((token) => layers.add(token.layer));
        return Array.from(layers);
    }


    async function combineMasks() {
        if (combining) return;

        const layers = getLayersWithSelections();
        if (layers.length !== 1) return;

        const layer = layers[0];
        const tokenIndices = $selectedTokensForCombining
            .filter((t) => t.layer === layer)
            .map((t) => t.tokenIdx);

        combining = true;
        try {
            const response = await api.combineMasks({
                layer,
                token_indices: tokenIndices,
                description,
            });

            // Add to combined masks store
            // $combinedMasks = [...$combinedMasks, {
            //     layer,
            //     tokenIndices,
            //     description: description || `Combined mask for ${layer}`,
            //     l0: response.mask_override.combined_mask.l0,
            //     createdAt: Date.now()
            // }];

            // Clear selections after successful combination
            $selectedTokensForCombining = [];
            $multiSelectMode = false;
            description = ""; // Reset description
        } catch (error) {
            console.error("Failed to combine masks:", error);
        } finally {
            combining = false;
        }
    }
</script>

<div class="mask-combine-panel">
    <div class="controls">
        <button
            class="mode-toggle"
            class:active={$multiSelectMode}
            on:click={toggleMultiSelectMode}
        >
            {$multiSelectMode ? "✓ Multi-Select Mode" : "Enable Multi-Select"}
        </button>
        <input type="text" bind:value={description} placeholder="Description" />

        {#if $multiSelectMode && $selectedTokensForCombining.length > 0}
            <button class="clear-btn" on:click={clearSelections}>
                Clear ({$selectedTokensForCombining.length})
            </button>

            {#if simulatedL0 !== null}
                <span class="l0-display">
                    L0: {simulating ? "..." : simulatedL0}
                </span>
            {/if}

            <button
                class="combine-btn"
                disabled={getLayersWithSelections().length !== 1 || combining}
                on:click={combineMasks}
            >
                {combining ? "Combining..." : "Combine Masks"}
            </button>

            {#if getLayersWithSelections().length > 1}
                <span class="warning"> ⚠️ Multiple layers selected </span>
            {/if}
        {/if}
    </div>
</div>

<style>
    .mask-combine-panel {
        padding: 0.75rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 1rem;
        background: white;
    }

    .controls {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }

    .mode-toggle {
        padding: 0.5rem 1rem;
        border: 2px solid #007bff;
        background: white;
        color: #007bff;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.2s;
    }

    .mode-toggle:hover {
        background: #f0f8ff;
    }

    .mode-toggle.active {
        background: #007bff;
        color: white;
    }

    .clear-btn {
        padding: 0.5rem 1rem;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .clear-btn:hover {
        background: #d32f2f;
    }

    .combine-btn {
        padding: 0.5rem 1rem;
        background: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }

    .combine-btn:hover:not(:disabled) {
        background: #45a049;
    }

    .combine-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .warning {
        padding: 0.25rem 0.5rem;
        color: #856404;
        font-size: 0.85rem;
    }

    .l0-display {
        padding: 0.5rem 1rem;
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 4px;
        color: #1976d2;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
