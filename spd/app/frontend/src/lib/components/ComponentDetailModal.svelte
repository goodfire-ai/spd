<script context="module">
    export type PopupData = {
        token: string;
        tokenIdx: number;
        layerIdx: number;
        layerName: string;
        tokenCIs: MatrixCausalImportances;
    };
</script>

<script lang="ts">
    import type { ClusterRunDTO, MatrixCausalImportances } from "$lib/api";
    import * as api from "$lib/api";
    import ComponentCard from "./ComponentCard.svelte";
    import HorizontalVirtualList from "./HorizontalVirtualList.svelte";
    import { onMount } from "svelte";

    export let onClose: () => void;
    export let toggleComponent: (
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ) => void;
    export let isComponentDisabled: (
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ) => boolean;

    export let cluster: ClusterRunDTO;
    export let dashboard: api.ClusterDashboardResponse;
    export let popupData: PopupData;

    type ComponentItem = {
        componentIdx: number;
        subcomponentCis: number[];
        componentAggCi: number;
    };

    $: componentItems = (() => {
        const groups: number[][] =
            cluster.clustering_shape.module_component_groups[popupData.layerName];
        const componentItems = groups.map<ComponentItem>((subcomponent_group, componentIdx) => ({
            componentIdx,
            subcomponentCis: subcomponent_group.map(
                (subcomponentIdx) => popupData.tokenCIs.subcomponent_cis[subcomponentIdx]
            ),
            componentAggCi: popupData.tokenCIs.component_agg_cis[componentIdx]
        }));

        componentItems.sort((a, b) => b.componentAggCi - a.componentAggCi || b.subcomponentCis.length - a.subcomponentCis.length);

        return componentItems;
    })();

    type ComponentExample = {
        textHash: string;
        rawText: string;
        offsetMapping: [number, number][];
        activations: number[];
    };


    // Build activation examples from dashboard data (like ClusterDashboardBody)
    $: textSampleLookup = Object.fromEntries(
        dashboard.text_samples.map((sample) => [
            sample.text_hash,
            { full_text: sample.full_text, tokens: sample.tokens }
        ])
    );

    function buildOffsets(tokens: string[]): [number, number][] {
        const offsets: [number, number][] = [];
        let cursor = 0;
        for (const token of tokens) {
            offsets.push([cursor, cursor + token.length]);
            cursor += token.length;
        }
        return offsets;
    }

    function normalizeActivations(values: number[], length: number): number[] {
        if (!values.length) return new Array(length).fill(0);
        const slice = values.slice(0, length);
        const max = Math.max(...slice.map((v) => Math.max(v, 0)), 1e-6);
        return slice.map((v) => Math.max(v, 0) / max);
    }

    // Map from componentIdx to cluster
    $: componentToClusterMap = new Map(
        dashboard.clusters.flatMap((cluster) =>
            cluster.components.map((comp) => [comp.index, cluster.cluster_hash])
        )
    );

    function buildExamplesForComponent(componentIdx: number): ComponentExample[] {
        const clusterHash = componentToClusterMap.get(componentIdx);
        if (!clusterHash) return [];

        const cluster = dashboard.clusters.find((c) => c.cluster_hash === clusterHash);
        if (!cluster?.criterion_samples?.["max_activation-max-16"]) return [];

        const hashes = cluster.criterion_samples["max_activation-max-16"];
        const activations = dashboard.activation_batch.activations;

        const examples: ComponentExample[] = [];
        for (const textHash of hashes.slice(0, 5)) {
            const activationHash = `${clusterHash}:${textHash}`;
            const idx = dashboard.activations_map[activationHash];
            if (typeof idx !== "number") continue;

            const sample = textSampleLookup[textHash];
            if (!sample?.tokens) continue;

            const activationValues = activations[idx];
            if (!activationValues || sample.tokens.length !== activationValues.length) continue;
            if (sample.tokens.length === 0) continue;

            const rawText = sample.tokens.join("");
            const offsets = buildOffsets(sample.tokens);
            const normalized = normalizeActivations(activationValues, offsets.length);

            examples.push({
                textHash,
                rawText,
                offsetMapping: offsets,
                activations: normalized
            });
        }

        return examples;
    }

    $: activationContextsMap = new Map(
        componentItems.map((item) => [
            item.componentIdx,
            buildExamplesForComponent(item.componentIdx)
        ])
    );

    function getAllComponentIndices(): number[] {
        // this is a little silly lol
        return cluster.clustering_shape.module_component_assignments[popupData.layerName].map(
            (_, idx) => idx
        );
    }

    function areAllComponentsDisabled(): boolean {
        if (!popupData) return false;
        const allIndices = getAllComponentIndices();
        return allIndices.every((idx) =>
            isComponentDisabled(popupData.layerName, popupData.tokenIdx, idx)
        );
    }

    function toggleAllComponents() {
        if (!popupData) return;
        const allIndices = getAllComponentIndices();
        const shouldDisable = !areAllComponentsDisabled();

        for (const componentIdx of allIndices) {
            const isCurrentlyDisabled = isComponentDisabled(
                popupData.layerName,
                popupData.tokenIdx,
                componentIdx
            );
            if (
                (shouldDisable && !isCurrentlyDisabled) ||
                (!shouldDisable && isCurrentlyDisabled)
            ) {
                toggleComponent(popupData.layerName, popupData.tokenIdx, componentIdx);
            }
        }
    }
</script>

{#if popupData}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="popup-overlay" on:click={onClose}>
        <div class="popup-modal" on:click|stopPropagation>
            <div class="popup-content">
                <div class="popup-info">
                    <p>
                        <strong>Token:</strong> "{popupData.token}" (position {popupData.tokenIdx})
                    </p>
                    <p><strong>Layer:</strong> {popupData.layerName}</p>
                    <p>
                        <strong>Subcomponents L0:</strong>
                        {popupData.tokenCIs.subcomponent_cis_sparse.l0}
                    </p>
                    <p>
                        <strong>Total Components:</strong>
                        {componentItems.length}
                    </p>
                    <p>
                        <strong>Component L0:</strong>
                        {popupData.tokenCIs.component_agg_cis.reduce(
                            (acc, val) => acc + (val > 0.0 ? 1 : 0),
                            0
                        )}
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
                    <div class="components-grid-container">
                        <HorizontalVirtualList
                            items={componentItems}
                            itemWidth={616}
                            buffer={2}
                            getKey={(item) => item.componentIdx}
                            let:item
                        >
                            <ComponentCard
                                componentIdx={item.componentIdx}
                                subcomponentCis={item.subcomponentCis}
                                componentAggCi={item.componentAggCi}
                                layer={popupData.layerName}
                                tokenIdx={popupData.tokenIdx}
                                examples={activationContextsMap.get(item.componentIdx)}
                                toggle={() => {
                                    toggleComponent(
                                        popupData.layerName,
                                        popupData.tokenIdx,
                                        item.componentIdx
                                    );
                                }}
                            />
                        </HorizontalVirtualList>
                    </div>
                </div>
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

    .components-grid-container {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fafafa;
    }
</style>
