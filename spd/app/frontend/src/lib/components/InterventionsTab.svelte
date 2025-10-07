<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import { onMount } from "svelte";
    import * as api from "$lib/api";
    import type {
        RunPromptResponse,
        ComponentMask,
        MatrixCausalImportances,
        AvailablePrompt,
        ClusterDashboardResponse
    } from "$lib/api";
    import {
        ablationComponentMask,
        ablationResults,
        promptWorkspaces,
        currentWorkspaceIndex,
        type PromptWorkspace
    } from "$lib/stores/componentState";

    import ComponentHeatmap from "$lib/components/ComponentHeatmap.svelte";
    import DisabledComponentsPanel from "$lib/components/DisabledComponentsPanel.svelte";
    import ComponentDetailModal, {
        type PopupData
    } from "$lib/components/ComponentDetailModal.svelte";
    import OriginalPredictions from "$lib/components/OriginalPredictions.svelte";
    import AblationPredictions from "$lib/components/AblationPredictions.svelte";
    import SavedMasksPanel from "$lib/components/SavedMasksPanel.svelte";

    export let cluster_run: api.ClusterRunDTO;
    export let iteration: number;

    let isLoading = false;
    let result: RunPromptResponse | null = null;
    let currentPromptId: string | null = null;
    let savedMasksPanel: SavedMasksPanel;
    let availablePrompts: AvailablePrompt[] | null = null;
    let showAvailablePrompts = false;
    let currentAblationPage = 0;

    let popupData: PopupData | null = null;
    let dashboard: ClusterDashboardResponse | null = null;

    async function loadAvailablePrompts() {
        try {
            availablePrompts = await api.getAvailablePrompts();
        } catch (error: any) {
            console.error("Failed to load prompts:", error.message);
        }
    }

    async function loadDashboard() {
        dashboard = await api.getClusterDashboardData({
            iteration,
            n_samples: 16,
            n_batches: 2,
            batch_size: 64,
            context_length: 64
        });
    }

    function toggleAvailablePrompts() {
        showAvailablePrompts = !showAvailablePrompts;
    }

    // Helper functions for workspace management
    function createNewWorkspace(promptData: RunPromptResponse): PromptWorkspace {
        const newMask: ComponentMask = {};
        for (const layer of promptData.layer_cis) {
            newMask[layer.module] = promptData.prompt_tokens.map(() => []);
        }

        return {
            promptId: promptData.prompt_id,
            promptData,
            ablationResults: [],
            runAblation: newMask
        };
    }

    function switchToWorkspace(index: number) {
        if (index >= 0 && index < $promptWorkspaces.length) {
            $currentWorkspaceIndex = index;
            const workspace = $promptWorkspaces[index];
            result = workspace.promptData;
            currentPromptId = workspace.promptId;
            $ablationComponentMask = workspace.runAblation;
            $ablationResults = workspace.ablationResults;
        }
    }

    function closeWorkspace(index: number) {
        $promptWorkspaces = $promptWorkspaces.filter((_, i) => i !== index);

        if ($promptWorkspaces.length === 0) {
            // No workspaces left
            result = null;
            currentPromptId = null;
            $currentWorkspaceIndex = 0;
        } else if (index <= $currentWorkspaceIndex) {
            // Adjust current index if necessary
            const newIndex = Math.max(0, $currentWorkspaceIndex - 1);
            $currentWorkspaceIndex = newIndex;
            switchToWorkspace(newIndex);
        }
    }

    async function runPromptByIndex(datasetIndex: number) {
        isLoading = true;
        try {
            const promptData = await api.runPromptByIndex(datasetIndex);
            const newWorkspace = createNewWorkspace(promptData);

            // Add new workspace and switch to it
            $promptWorkspaces = [...$promptWorkspaces, newWorkspace];
            $currentWorkspaceIndex = $promptWorkspaces.length - 1;
            switchToWorkspace($currentWorkspaceIndex);
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    async function applyMaskAsAblation(maskId: string) {
        if (!result || !currentPromptId) return;

        isLoading = true;
        try {
            const maskResult = await api.applyMaskAsAblation(currentPromptId, maskId);

            // Get mask info for display
            const masks = await api.getMaskOverrides();
            const appliedMask = masks.find((m) => m.id === maskId);

            // Add as an ablation result with description
            $ablationResults = [
                ...$ablationResults,
                {
                    tokenLogits: maskResult.token_logits,
                    applied_mask: {}, // The mask was applied uniformly to all tokens
                    id: Date.now(),
                    maskOverride: appliedMask, // Store mask info for display
                    ablationStats: maskResult.ablation_stats
                }
            ];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    // Refresh saved masks after creating a new one
    export function refreshSavedMasks() {
        if (savedMasksPanel) {
            savedMasksPanel.loadMasks();
        }
    }

    function initializeRunAblation() {
        if (!result) return;
        const newMask: ComponentMask = {};
        for (const layer of result.layer_cis) {
            newMask[layer.module] = result.prompt_tokens.map(() => []);
        }
        $ablationComponentMask = newMask;
    }

    function toggleComponentDisabled(layerName: string, tokenIdx: number, componentIdx: number) {
        if (!$ablationComponentMask[layerName]) {
            $ablationComponentMask[layerName] = result!.prompt_tokens.map(() => []);
        }

        const disabledComponents = $ablationComponentMask[layerName][tokenIdx]; const existingIdx = disabledComponents.indexOf(componentIdx);

        if (existingIdx === -1) {
            disabledComponents.push(componentIdx);
        } else {
            disabledComponents.splice(existingIdx, 1);
        }

        $ablationComponentMask = { ...$ablationComponentMask };
    }

    function isComponentDisabled(
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ): boolean {
        return $ablationComponentMask[layerName][tokenIdx].includes(componentIdx);
    }

    async function sendAblation() {
        if (!result || !currentPromptId) return;

        isLoading = true;
        try {
            const data = await api.ablateComponents(currentPromptId, $ablationComponentMask);

            const deepCopyMask: ComponentMask = {};
            for (const [layerName, tokenArrays] of Object.entries($ablationComponentMask)) {
                deepCopyMask[layerName] = tokenArrays.map((tokenMask) => [...tokenMask]);
            }

            $ablationResults = [
                ...$ablationResults,
                {
                    tokenLogits: data.token_logits,
                    applied_mask: deepCopyMask,
                    id: Date.now(),
                    ablationStats: data.ablation_stats
                }
            ];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    async function runRandomPrompt() {
        if (!availablePrompts) {
            console.error("No prompts available");
            return;
        }
        runPromptByIndex(Math.floor(Math.random() * availablePrompts.length));
    }

    function openPopup(
        token: string,
        tokenIdx: number,
        layerIdx: number,
        layerName: string,
        tokenCIs: MatrixCausalImportances
    ) {
        popupData = { token, tokenIdx, layerIdx, layerName, tokenCIs };
    }

    function closePopup() {
        popupData = null;
    }

    $: if (result) {
        initializeRunAblation();
    }

    onMount(() => {
        loadAvailablePrompts();
        loadDashboard();
    });
</script>

<div class="tab-content">
    <SavedMasksPanel bind:this={savedMasksPanel} onApplyMask={applyMaskAsAblation} />
    <div class="workspace-navigation">
        <div class="workspace-header">
            <h3>Prompt Workspaces</h3>
            {#if showAvailablePrompts}
                <button class="add-prompt-btn" on:click={toggleAvailablePrompts}>Cancel</button>
            {:else}
                <button class="add-prompt-btn" on:click={runRandomPrompt}>+ Random Prompt</button>
                <button class="add-prompt-btn" on:click={toggleAvailablePrompts}
                    >+ Add Prompt</button
                >
            {/if}
        </div>

        {#if showAvailablePrompts}
            <div class="available-prompts-dropdown">
                {#if availablePrompts == null}
                    <p>Loading prompts...</p>
                {:else}
                    <div class="prompt-list">
                        {#each availablePrompts as prompt, i}
                            <button
                                class="prompt-button"
                                on:click={() => {
                                    runPromptByIndex(prompt.index);
                                    showAvailablePrompts = false;
                                }}
                                disabled={isLoading}
                            >
                                <span class="prompt-number">#{i + 1}</span>
                                <span class="prompt-text"
                                    >{prompt.full_text.slice(0, 40)}{prompt.full_text.length > 40
                                        ? "..."
                                        : ""}</span
                                >
                            </button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/if}

        {#if $promptWorkspaces.length > 0}
            <div class="workspace-list">
                {#each $promptWorkspaces as workspace, i}
                    <button
                        class="workspace-item"
                        class:active={i === $currentWorkspaceIndex}
                        on:click={() => switchToWorkspace(i)}
                    >
                        <span class="workspace-number">#{i + 1}</span>
                        <span class="workspace-text">
                            {workspace.promptData.prompt_tokens.slice(0, 8).join(" ")}...
                        </span>
                        <span
                            class="workspace-close"
                            on:click|stopPropagation={() => closeWorkspace(i)}>×</span
                        >
                    </button>
                {/each}
            </div>
        {:else}
            <div class="empty-workspaces">No prompts loaded. Click "Add Prompt" to start.</div>
        {/if}
    </div>

    <div class="main-layout">
        <div class="left-panel">
            {#if result && currentPromptId}
                <ComponentHeatmap
                    {result}
                    promptId={currentPromptId}
                    onCellPopop={openPopup}
                    on:maskCreated={refreshSavedMasks}
                />
                <DisabledComponentsPanel
                    promptTokens={result.prompt_tokens}
                    {isLoading}
                    onSendAblation={sendAblation}
                    onToggleComponent={toggleComponentDisabled}
                />
            {/if}
        </div>

        <!-- Right side: Scrollable predictions -->
        <div class="right-panel">
            {#if result && result.full_run_token_logits}
                <OriginalPredictions
                    tokenLogits={result.full_run_token_logits}
                    promptTokens={result.prompt_tokens}
                    title="Original Model Predictions"
                />
            {/if}

            {#if result && result.ci_masked_token_logits}
                <OriginalPredictions
                    tokenLogits={result.ci_masked_token_logits}
                    promptTokens={result.prompt_tokens}
                    title="Original <strong>CI Masked</strong> Model Predictions"
                />
            {/if}

            {#if result && $ablationResults.length > 0}
                <div class="ablation-results-container">
                    <div class="pagination-header">
                        <h3>Ablation Results ({$ablationResults.length} total)</h3>
                        <div class="pagination-controls">
                            <button
                                on:click={() => (currentAblationPage = Math.max(0, currentAblationPage - 1))}
                                disabled={currentAblationPage === 0}
                            >
                                ←
                            </button>
                            <span>{currentAblationPage + 1} / {$ablationResults.length}</span>
                            <button
                                on:click={() => (currentAblationPage = Math.min($ablationResults.length - 1, currentAblationPage + 1))}
                                disabled={currentAblationPage === $ablationResults.length - 1}
                            >
                                →
                            </button>
                        </div>
                    </div>
                    <AblationPredictions
                        tokenLogits={$ablationResults[currentAblationPage].tokenLogits}
                        promptTokens={result.prompt_tokens}
                        appliedMask={$ablationResults[currentAblationPage].applied_mask}
                        maskOverride={$ablationResults[currentAblationPage].maskOverride}
                        ablationStats={$ablationResults[currentAblationPage].ablationStats}
                    />
                </div>
            {/if}
        </div>
    </div>

    {#if popupData && dashboard}
        <ComponentDetailModal
            cluster={cluster_run}
            {popupData}
            {dashboard}
            onClose={closePopup}
            toggleComponent={toggleComponentDisabled}
            {isComponentDisabled}
        />
    {/if}
</div>

<style>
    .tab-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }

    .main-layout {
        display: flex;
        gap: 1rem;
        min-height: 70vh;
    }

    .left-panel {
        flex: 1;
        min-width: 0;
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
    }

    .right-panel {
        flex: 1;
        min-width: 0;
        overflow-y: auto;
        padding-right: 1rem;
    }

    .prompt-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
    }

    .prompt-button {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem;
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 6px;
        cursor: pointer;
        text-align: left;
        transition: all 0.2s;
        font-family: inherit;
    }

    .prompt-button:hover:not(:disabled) {
        border-color: #007bff;
        background: #f0f8ff;
    }

    .prompt-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }

    .prompt-number {
        font-weight: bold;
        color: #666;
        min-width: 2rem;
        margin-top: 0.1rem;
    }

    .prompt-text {
        flex: 1;
        font-size: 0.9rem;
        line-height: 1.4;
        color: #333;
    }

    .workspace-navigation {
        margin-bottom: 1rem;
        padding: 1rem;
        background: #e8f4f8;
        border: 1px solid #b3d9e8;
        border-radius: 8px;
    }

    .workspace-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .workspace-header h3 {
        margin: 0;
        font-size: 1.1rem;
        color: #333;
    }

    .add-prompt-btn {
        padding: 0.5rem 1rem;
        background: #28a745;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: 500;
        transition: background 0.2s;
    }

    .add-prompt-btn:hover {
        background: #218838;
    }

    .available-prompts-dropdown {
        margin-bottom: 1rem;
        padding: 1rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .workspace-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .workspace-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: white;
        border: 2px solid #b3d9e8;
        border-radius: 6px;
        cursor: pointer;
        text-align: left;
        transition: all 0.2s;
        font-family: inherit;
    }

    .workspace-item:hover {
        background: #f0f8ff;
        border-color: #007bff;
    }

    .workspace-item.active {
        background: #007bff;
        color: white;
        border-color: #0056b3;
    }

    .workspace-number {
        font-weight: bold;
        min-width: 2rem;
        color: #666;
    }

    .workspace-item.active .workspace-number {
        color: white;
    }

    .workspace-text {
        flex: 1;
        font-size: 0.9rem;
        line-height: 1.4;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .workspace-close {
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        padding: 0.1rem 0.3rem;
        border-radius: 50%;
        transition: background 0.2s;
        color: #666;
    }

    .workspace-item.active .workspace-close {
        color: white;
    }

    .workspace-close:hover {
        background: rgba(255, 0, 0, 0.1);
        color: red;
    }

    .workspace-item.active .workspace-close:hover {
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }

    .empty-workspaces {
        padding: 2rem;
        text-align: center;
        color: #6c757d;
        font-style: italic;
    }

    .ablation-results-container {
        margin-top: 1rem;
    }

    .pagination-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 6px;
    }

    .pagination-header h3 {
        margin: 0;
        font-size: 1rem;
        color: #333;
    }

    .pagination-controls {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .pagination-controls button {
        padding: 0.25rem 0.75rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
        transition: all 0.2s;
    }

    .pagination-controls button:hover:not(:disabled) {
        background: #007bff;
        color: white;
        border-color: #007bff;
    }

    .pagination-controls button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }

    .pagination-controls span {
        font-size: 0.9rem;
        color: #666;
        min-width: 4rem;
        text-align: center;
    }
</style>
