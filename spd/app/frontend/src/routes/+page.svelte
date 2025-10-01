<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import { onMount } from "svelte";
    import { api } from "$lib/api";
    import type {
        RunPromptResponse,
        ComponentMask,
        Status,
        MatrixCausalImportances
    } from "$lib/api";
    import {
        ablationComponentMask,
        popupData,
        ablationResults,
        promptWorkspaces,
        currentWorkspaceIndex,
        type PromptWorkspace
    } from "$lib/stores/componentState";

    import RunSelector from "$lib/components/RunSelector.svelte";
    import ComponentHeatmap from "$lib/components/ComponentHeatmap.svelte";
    import DisabledComponentsPanel from "$lib/components/DisabledComponentsPanel.svelte";
    import ComponentDetailModal from "$lib/components/ComponentDetailModal.svelte";
    import OriginalPredictions from "$lib/components/OriginalPredictions.svelte";
    import AblationPredictions from "$lib/components/AblationPredictions.svelte";
    import SavedMasksPanel from "$lib/components/SavedMasksPanel.svelte";
    import ActivationContextsTab from "$lib/components/ActivationContextsTab.svelte";

    let isLoading = false;
    let result: RunPromptResponse | null = null;
    let currentPromptId: string | null = null;
    let wandbRunId: string | null = null;
    let loadingRun = false;
    let savedMasksPanel: SavedMasksPanel;
    let availablePrompts: { index: number; text: string; full_text: string }[] = [];
    let showAvailablePrompts = false;
    let activeTab: "ablation" | "activation-contexts" = "ablation";

    async function loadAvailablePrompts() {
        try {
            availablePrompts = await api.getAvailablePrompts();
        } catch (error: any) {
            console.error("Failed to load prompts:", error.message);
        }
    }

    function toggleAvailablePrompts() {
        showAvailablePrompts = !showAvailablePrompts;
        if (showAvailablePrompts && availablePrompts.length === 0) {
            loadAvailablePrompts();
        }
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
                    maskOverride: appliedMask // Store mask info for display
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

        const disabledComponents = $ablationComponentMask[layerName][tokenIdx];
        const existingIdx = disabledComponents.indexOf(componentIdx);

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
                    id: Date.now()
                }
            ];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    function openPopup(
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        matrixCis: MatrixCausalImportances
    ) {
        $popupData = { token, tokenIdx, layer, layerIdx, matrixCis };
    }

    function closePopup() {
        $popupData = null;
    }

    $: if (result) {
        initializeRunAblation();
    }

    let status: Status | null = null;
    async function getStatus() {
        console.log("getting status");
        status = await api.getStatus();
        console.log("status", status);
    }

    $: wandbRunId = status?.run_id ?? null;

    onMount(() => {
        getStatus();
        loadAvailablePrompts();
    });
</script>

<main>
    <div class="container">
        <!-- Top-level controls -->
        <div class="top-controls">
            <RunSelector bind:loadingRun bind:wandbRunId {isLoading} />

            <div class="tab-navigation">
                <button
                    class="tab-button"
                    class:active={activeTab === "ablation"}
                    on:click={() => (activeTab = "ablation")}
                >
                    Component Ablation
                </button>
                <button
                    class="tab-button"
                    class:active={activeTab === "activation-contexts"}
                    on:click={() => (activeTab = "activation-contexts")}
                >
                    Activation Contexts
                </button>
            </div>
        </div>

        <div class:hidden={activeTab !== "ablation"}>
            <SavedMasksPanel bind:this={savedMasksPanel} onApplyMask={applyMaskAsAblation} />
            <div class="workspace-navigation">
                <div class="workspace-header">
                    <h3>Prompt Workspaces</h3>
                    <button class="add-prompt-btn" on:click={toggleAvailablePrompts}>
                        {showAvailablePrompts ? "Cancel" : "+ Add Prompt"}
                    </button>
                </div>

                {#if showAvailablePrompts}
                    <div class="available-prompts-dropdown">
                        {#if availablePrompts.length === 0}
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
                                        <span class="prompt-text">{prompt.text}</span>
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
                    <div class="empty-workspaces">
                        No prompts loaded. Click "Add Prompt" to start.
                    </div>
                {/if}
            </div>

            <div class="main-layout">
                <div class="left-panel">
                    {#if result && currentPromptId}
                        <ComponentHeatmap
                            {result}
                            promptId={currentPromptId}
                            onCellClick={openPopup}
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

                    {#if result}
                        {#each $ablationResults as ablationResult}
                            <AblationPredictions
                                tokenLogits={ablationResult.tokenLogits}
                                promptTokens={result.prompt_tokens}
                                appliedMask={ablationResult.applied_mask}
                                maskOverride={ablationResult.maskOverride}
                            />
                        {/each}
                    {/if}
                </div>
            </div>
        </div>
        <div class:hidden={activeTab !== "activation-contexts"}>
            <!-- Activation Contexts Tab Content -->
            {#if status}
                <div class="activation-contexts-container">
                    <ActivationContextsTab availableComponentLayers={status.component_layers} />
                </div>
            {/if}
        </div>

        <ComponentDetailModal
            onClose={closePopup}
            onToggleComponent={toggleComponentDisabled}
            {isComponentDisabled}
        />
    </div>
</main>

<style>
    main {
        padding: 0.5rem;
        font-family:
            system-ui,
            -apple-system,
            sans-serif;
    }

    .container {
        max-width: none;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .top-controls {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .main-layout {
        display: flex;
        gap: 1rem;
        min-height: 70vh;
    }

    .left-panel {
        padding: 0.5rem;
        flex: 0 0 50%;
        max-width: 50%;
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
    }

    .right-panel {
        flex: 1;
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

    .tab-navigation {
        display: flex;
        gap: 0.5rem;
        padding: 0 1rem;
        background: #f8f9fa;
        border-bottom: 2px solid #dee2e6;
    }

    .tab-button {
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500;
        color: #6c757d;
        transition: all 0.2s;
        position: relative;
        top: 2px;
    }

    .tab-button:hover {
        color: #007bff;
    }

    .tab-button.active {
        color: #007bff;
        border-bottom-color: #007bff;
        background: white;
    }

    .main-layout.hidden {
        display: none;
    }

    .activation-contexts-container {
        padding: 1rem;
        background: white;
        border-radius: 8px;
        min-height: 60vh;
    }
</style>
