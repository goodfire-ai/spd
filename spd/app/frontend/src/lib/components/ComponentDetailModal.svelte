<script lang="ts">
    import { popupData, runAblation } from "$lib/stores/componentState";
    import { api } from "$lib/api";
    import type { CosineSimilarityData } from "$lib/api";
    import CosineSimilarityPlot from "./CosineSimilarityPlot.svelte";

    export let onClose: () => void;
    export let onToggleComponent: (layerName: string, tokenIdx: number, componentIdx: number) => void;
    export let isComponentDisabled: (layerName: string, tokenIdx: number, componentIdx: number) => boolean;

    let similarityData: CosineSimilarityData | null = null;
    let loadingSimilarities = false;

    // Load cosine similarities when popup data changes
    $: if ($popupData) {
        loadCosineSimilarities($popupData.layer, $popupData.tokenIdx);
    }

    async function loadCosineSimilarities(layer: string, tokenIdx: number) {
        loadingSimilarities = true;
        try {
            similarityData = await api.getCosineSimilarities(layer, tokenIdx);
        } catch (error) {
            console.error("Failed to load cosine similarities:", error);
            similarityData = null;
        }
        loadingSimilarities = false;
    }

    function getColorFromCI(ci: number): string {
        const whiteAmount = Math.round((1 - ci) * 255);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    }

    function getAllComponentIndices(): number[] {
        if (!$popupData) return [];
        return $popupData.tokenCis.indices;
    }

    function areAllComponentsDisabled(): boolean {
        if (!$popupData) return false;
        const allIndices = getAllComponentIndices();
        return allIndices.every((idx) =>
            isComponentDisabled($popupData!.layer, $popupData!.tokenIdx, idx)
        );
    }

    function toggleAllComponents() {
        if (!$popupData) return;
        const allIndices = getAllComponentIndices();
        const shouldDisable = !areAllComponentsDisabled();

        for (const componentIdx of allIndices) {
            const isCurrentlyDisabled = isComponentDisabled(
                $popupData.layer,
                $popupData.tokenIdx,
                componentIdx
            );
            if (shouldDisable && !isCurrentlyDisabled) {
                onToggleComponent($popupData.layer, $popupData.tokenIdx, componentIdx);
            } else if (!shouldDisable && isCurrentlyDisabled) {
                onToggleComponent($popupData.layer, $popupData.tokenIdx, componentIdx);
            }
        }
    }

    // Make this reactive to $runAblation changes
    $: disabledComponentIndices = (() => {
        if (!$popupData) return [];
        // Access $runAblation to trigger reactivity
        const ablations = $runAblation;
        const allIndices = getAllComponentIndices();
        return allIndices.filter((idx) => {
            const layerAblations = ablations[$popupData.layer];
            if (!layerAblations || !layerAblations[$popupData.tokenIdx]) return false;
            return layerAblations[$popupData.tokenIdx].includes(idx);
        });
    })();
</script>

{#if $popupData}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="popup-overlay" on:click={onClose}>
        <div class="popup-modal" on:click|stopPropagation>
            <div class="popup-content">
                <div class="popup-info">
                    <p>
                        <strong>Token:</strong> "{$popupData.token}" (position {$popupData.tokenIdx})
                    </p>
                    <p><strong>Layer:</strong> {$popupData.layer}</p>
                    <p><strong>L0 (Non-zero components):</strong> {$popupData.tokenCis.l0}</p>
                    <p>
                        <strong>Vector Length:</strong>
                        {$popupData.tokenCis.values.length}
                    </p>
                </div>
                <div class="vector-display">
                    <div class="vector-controls">
                        <h4>Component Values:</h4>
                        <label class="select-all-label">
                            <input
                                type="checkbox"
                                checked={areAllComponentsDisabled()}
                                on:change={toggleAllComponents}
                            />
                            Select All
                        </label>
                    </div>
                    <div class="vector-grid">
                        {#each $popupData.tokenCis.values as value, idx}
                            <div
                                class="vector-item"
                                style="--item-bg-color: {getColorFromCI(value / 3)}"
                                class:disabled={isComponentDisabled(
                                    $popupData.layer,
                                    $popupData.tokenIdx,
                                    $popupData.tokenCis.indices[idx]
                                )}
                                on:click={() => {
                                    if ($popupData) {
                                        onToggleComponent(
                                            $popupData.layer,
                                            $popupData.tokenIdx,
                                            $popupData.tokenCis.indices[idx]
                                        );
                                    }
                                }}
                            >
                                <span class="component-idx"
                                    >{$popupData.tokenCis.indices[idx]}:</span
                                >
                                <span class="component-value">{value.toFixed(4)}</span>
                            </div>
                        {/each}
                    </div>
                </div>

                <!-- Cosine Similarity Plots -->
                {#if loadingSimilarities}
                    <div class="loading-similarities">Loading similarity data...</div>
                {:else if similarityData}
                    <div class="similarity-plots">
                        <h3>Pairwise Cosine Similarities</h3>
                        <div class="plots-container">
                            <CosineSimilarityPlot
                                title="Input Singular Vectors"
                                data={similarityData.input_singular_vectors}
                                indices={similarityData.component_indices}
                                disabledIndices={disabledComponentIndices}
                            />
                            <CosineSimilarityPlot
                                title="Output Singular Vectors"
                                data={similarityData.output_singular_vectors}
                                indices={similarityData.component_indices}
                                disabledIndices={disabledComponentIndices}
                            />
                        </div>
                    </div>
                {/if}
            </div>
        </div>
    </div>
{/if}

<style>
    .popup-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }

    .popup-modal {
        background: white;
        border-radius: 8px;
        padding: 0;
        max-width: 600px;
        max-height: 80vh;
        width: 90%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    }

    .popup-content {
        padding: 1.5rem;
        overflow-y: auto;
        max-height: calc(80vh - 80px);
    }

    .popup-info {
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 4px;
    }

    .popup-info p {
        margin: 0.5rem 0;
        color: #555;
    }

    .vector-display h4 {
        margin: 0 0 1rem 0;
        color: #333;
    }

    .vector-controls {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .select-all-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: #333;
        cursor: pointer;
    }

    .select-all-label input[type="checkbox"] {
        cursor: pointer;
    }

    .vector-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #eee;
        padding: 1rem;
        border-radius: 4px;
    }

    .vector-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: background-color 0.2s ease;
        background-color: var(--item-bg-color, #e5e2ff);
    }

    .vector-item.disabled {
        background-color: #ff4444 !important;
        opacity: 0.7;
    }

    .component-idx {
        color: #666;
        font-weight: bold;
        flex-shrink: 0;
    }

    .component-value {
        font-weight: bold;
        text-align: right;
        flex-grow: 1;
        margin: 0 0.5rem;
    }

    .similarity-plots {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
    }

    .similarity-plots h3 {
        margin: 0 0 1rem 0;
        color: #333;
        font-size: 1rem;
    }

    .plots-container {
        display: flex;
        gap: 2rem;
        justify-content: space-around;
        flex-wrap: wrap;
    }

    .loading-similarities {
        margin-top: 1rem;
        padding: 1rem;
        text-align: center;
        color: #666;
        font-style: italic;
    }
</style>