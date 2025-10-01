<script lang="ts">
    import { ablationComponentMask, popupData } from "$lib/stores/componentState";
    import ComponentCard from "./ComponentCard.svelte";
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

    function getAllComponentIndices(): number[] {
        if (!$popupData) return [];
        return $popupData.matrixCis.components.map((component) => component.index);
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
                            <ComponentCard
                                {component}
                                componentAggCi={$popupData.matrixCis.component_agg_cis[
                                    component.index
                                ]}
                                layer={$popupData.layer}
                                tokenIdx={$popupData.tokenIdx}
                                onToggle={() => {
                                    if ($popupData) {
                                        onToggleComponent(
                                            $popupData.layer,
                                            $popupData.tokenIdx,
                                            component.index
                                        );
                                    }
                                }}
                            />
                        {/each}
                    </div>
                </div>
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
