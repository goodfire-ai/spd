<script lang="ts">
    import type { CosineSimilarityData } from "$lib/api";
    import { api } from "$lib/api";
    import { ablationComponentMask, popupData } from "$lib/stores/componentState";
    import CosineSimilarityPlot from "./CosineSimilarityPlot.svelte";
    // import ActivationContexts from "./ActivationContexts.svelte";

    export let onClose: () => void;
    export let onToggleComponent: (
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ) => void;
    export let isComponentDisabled: (
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ) => boolean;
    export let promptId: string | null = null;

    let similarityData: CosineSimilarityData | null = null;
    let loadingSimilarities = false;

    // Load cosine similarities when popup data changes
    $: if ($popupData && promptId) {
        loadCosineSimilarities(promptId, $popupData.layer, $popupData.tokenIdx);
    }

    async function loadCosineSimilarities(promptId: string, layer: string, tokenIdx: number) {
        loadingSimilarities = true;
        try {
            similarityData = await api.getCosineSimilarities(promptId, layer, tokenIdx);
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
        return $popupData.matrixCis.component_agg_cis;
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
        const ablations = $ablationComponentMask;
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
                    <p>
                        <strong>Subcomponents L0:</strong>
                        {$popupData.matrixCis.subcomponent_cis.l0}
                    </p>
                </div>
                <div class="components-section">
                    <div class="section-header">
                        <h4>Components</h4>
                        <label class="select-all-label">
                            <input
                                type="checkbox"
                                checked={areAllComponentsDisabled()}
                                on:change={toggleAllComponents}
                            />
                            Ablate All
                        </label>
                    </div>
                    <div class="components-grid">
                        {#each $popupData.matrixCis.components as component}
                            <div
                                class="component-card"
                                style="--component-bg: {getColorFromCI(
                                    $popupData.matrixCis.component_agg_cis[component.index] / 2
                                )}"
                                class:disabled={isComponentDisabled(
                                    $popupData.layer,
                                    $popupData.tokenIdx,
                                    component.index
                                )}
                                on:click={() => {
                                    if ($popupData) {
                                        onToggleComponent(
                                            $popupData.layer,
                                            $popupData.tokenIdx,
                                            component.index
                                        );
                                    }
                                }}
                            >
                                <div class="component-header">
                                    <span class="component-index">#{component.index}</span>
                                    <span class="component-ci"
                                        >{$popupData.matrixCis.component_agg_cis[
                                            component.index
                                        ].toFixed(4)}</span
                                    >
                                    <span class="component-rank"
                                        >{$popupData.matrixCis.components[component.index]
                                            .subcomponent_indices.length}</span
                                    >
                                </div>
                                <!-- {#if component.subcomponent_indices.length > 0}
                                    <div class="subcomponents">
                                        {#each component.subcomponent_indices as subIdx}
                                            <span class="subcomponent-badge" >{subIdx}</span >
                                        {/each}
                                    </div>
                                {/if} -->
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

                <!-- Activation Contexts Section -->
                <!-- {#if $popupData.tokenCis.indices.length > 0}
                    <div class="activation-contexts-section">
                        <h3>Activation Examples</h3>
                        <p class="section-description">
                            Examples of prompts where these components activate:
                        </p>
                        {#each $popupData.tokenCis.indices.slice(0, 3) as componentIdx}
                            <div class="component-activation">
                                <h4>Component {componentIdx}</h4>
                                <ActivationContexts
                                    componentId={componentIdx}
                                    layer={$popupData.layer}
                                    compact={true}
                                />
                            </div>
                        {/each}
                    </div>
                {/if} -->
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
        padding: 1.5rem;
        max-height: 80vh;
        width: 90%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        overflow-y: auto;
    }

    .popup-content {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .popup-info {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 4px;
    }

    .popup-info p {
        margin: 0.5rem 0;
        color: #555;
    }


    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .section-header h4 {
        margin: 0;
        color: #333;
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

    .components-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 0.75rem;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fafafa;
    }

    .component-card {
        background-color: var(--component-bg);
        border-radius: 6px;
        padding: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 2px solid transparent;
    }

    .component-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-color: rgba(0, 0, 0, 0.1);
    }

    .component-card.disabled {
        background-color: #ff6b6b !important;
        opacity: 0.8;
    }

    .component-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .component-index {
        font-weight: 600;
        color: #333;
        font-size: 0.95rem;
    }

    .component-ci {
        font-weight: 700;
        color: #1a1a1a;
        font-size: 0.9rem;
        font-family: "Monaco", "Courier New", monospace;
    }

    .component-rank {
        font-weight: 700;
        color: #1a1a1a;
        font-size: 0.9rem;
        font-family: "Monaco", "Courier New", monospace;
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

    .activation-contexts-section {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
    }

    .activation-contexts-section h3 {
        margin: 0 0 0.5rem 0;
        color: #333;
        font-size: 1rem;
    }

    .section-description {
        margin: 0 0 1rem 0;
        color: #666;
        font-size: 0.9rem;
    }

    .component-activation {
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #e9ecef;
    }

    .component-activation h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: #495057;
        font-weight: 600;
    }
</style>
