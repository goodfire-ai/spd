<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    import { api } from "$lib/api";
    import type { OutputTokenLogit, RunPromptResponse, ComponentMask } from "$lib/api";
    let prompt =
        "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked";
    let isLoading = false;
    let result: RunPromptResponse | null = null;
    let wandbRunId = "";
    let loadingRun = false;

    // Preset run options
    const presetRuns = [
        { id: "ry05f67a", label: "Run ry05f67a" },
        { id: "6a7en259", label: "Run 6a7en259" }
    ];

    async function runPrompt() {
        if (!prompt.trim()) return;

        isLoading = true;
        try {
            result = await api.runPrompt(prompt);
            modificationResults = [];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    async function loadRun() {
        if (!wandbRunId.trim()) return;

        loadingRun = true;
        try {
            await api.loadRun(wandbRunId);
            // After loading, run the current prompt to see the results
        } catch (error: any) {
            console.error(`Error loading run: ${error.message}`);
            alert(`Failed to load run: ${error.message}`);
        }
        loadingRun = false;

        await runPrompt();
    }

    // Run modification state - tracks disabled components
    let runModification: ComponentMask = {};

    // Array of modification results (for stacking)
    let modificationResults: Array<{
        tokenLogits: OutputTokenLogit[][];
        applied_mask: ComponentMask;
        id: number;
    }> = [];

    // Popup state
    let popupData: {
        token: string;
        tokenIdx: number;
        layer: string;
        layerIdx: number;
        tokenCi: {
            l0: number;
            component_cis: number[];
            indices: number[];
        };
    } | null = null;

    // Selection state for popup

    // Calculate global max for normalization
    let getGlobalMax = () => {
        if (!result) return 0;
        return Math.max(
            ...result!.layer_cis.flatMap((layer) => layer.token_cis.map((tokenCIs) => tokenCIs.l0))
        );
    };

    // Reactive function that depends on runModification
    $: getColorFroml0 = (l0: number, layerName: string, tokenIdx: number) => {
        const intensity = Math.max(0, Math.min(1, l0 / getGlobalMax()));

        // Calculate percentage of disabled components for this cell
        const disabledComponents = runModification[layerName]?.[tokenIdx]?.length ?? 0;
        const totalComponents = l0; // L0 is the count of non-zero components
        const disabledRatio = totalComponents > 0 ? disabledComponents / totalComponents : 0;

        // Base blue color (white to blue based on L0 intensity)
        const whiteAmount = Math.round((1 - intensity) * 255);
        const baseColor = `rgb(${whiteAmount}, ${whiteAmount}, 255)`;

        // If no components are disabled, return base color
        if (disabledRatio === 0) {
            return baseColor;
        }

        // Create a gradient overlay: red progress bar from left, base color for remainder
        const disabledPercent = Math.round(disabledRatio * 100);
        return `linear-gradient(to right, #ff4444 0%, #ff4444 ${disabledPercent}%, ${baseColor} ${disabledPercent}%, ${baseColor} 100%)`;
    };

    function getColorFromCI(ci: number): string {
        const whiteAmount = Math.round((1 - ci) * 255);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    }

    function openPopup(
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        token_ci: { l0: number; component_cis: number[]; indices: number[] }
    ) {
        popupData = { token, tokenIdx, layer, layerIdx, tokenCi: token_ci };
    }

    function closePopup() {
        popupData = null;
    }

    // Component disabling functions
    function initializeRunModification() {
        if (!result) return;

        runModification = {};
        for (const layer of result.layer_cis) {
            runModification[layer.module] = result.prompt_tokens.map(() => []);
        }
    }

    function toggleComponentDisabled(layerName: string, tokenIdx: number, componentIdx: number) {
        if (!runModification[layerName]) {
            runModification[layerName] = result!.prompt_tokens.map(() => []);
        }

        const disabledComponents = runModification[layerName][tokenIdx];
        const existingIdx = disabledComponents.indexOf(componentIdx);

        if (existingIdx === -1) {
            // Add to disabled list
            disabledComponents.push(componentIdx);
        } else {
            // Remove from disabled list
            disabledComponents.splice(existingIdx, 1);
        }

        // Trigger reactivity
        runModification = { ...runModification };
    }

    function isComponentDisabled(
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ): boolean {
        return runModification[layerName]?.[tokenIdx]?.includes(componentIdx) ?? false;
    }

    // Get all component indices for current popup
    function getAllComponentIndices(): number[] {
        if (!popupData) return [];
        return popupData.tokenCi.indices;
    }

    // Check if all components are disabled
    function areAllComponentsDisabled(): boolean {
        if (!popupData) return false;
        const allIndices = getAllComponentIndices();
        return allIndices.every((idx) =>
            isComponentDisabled(popupData!.layer, popupData!.tokenIdx, idx)
        );
    }

    // Toggle all components
    function toggleAllComponents() {
        if (!popupData) return;
        const allIndices = getAllComponentIndices();
        const shouldDisable = !areAllComponentsDisabled();

        for (const componentIdx of allIndices) {
            const isCurrentlyDisabled = isComponentDisabled(
                popupData.layer,
                popupData.tokenIdx,
                componentIdx
            );
            if (shouldDisable && !isCurrentlyDisabled) {
                toggleComponentDisabled(popupData.layer, popupData.tokenIdx, componentIdx);
            } else if (!shouldDisable && isCurrentlyDisabled) {
                toggleComponentDisabled(popupData.layer, popupData.tokenIdx, componentIdx);
            }
        }
    }

    async function sendModification() {
        if (!result) return;

        isLoading = true;
        try {
            const data = await api.modifyComponents(prompt, runModification);

            // Append new modification result with proper deep copy
            const deepCopyMask: ComponentMask = {};
            for (const [layerName, tokenArrays] of Object.entries(runModification)) {
                deepCopyMask[layerName] = tokenArrays.map((tokenMask) => [...tokenMask]);
            }

            modificationResults = [
                ...modificationResults,
                {
                    tokenLogits: data.token_logits,
                    applied_mask: deepCopyMask,
                    id: Date.now() // Simple unique ID
                }
            ];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    // Initialize run modification when result changes
    $: if (result) {
        initializeRunModification();
    }

    // Scroll synchronization
    let isScrolling = false;

    function syncScroll(event: Event) {
        if (isScrolling) return;

        const target = event.target as HTMLElement;
        const scrollLeft = target.scrollLeft;

        isScrolling = true;

        // Find all logits display containers and sync their scroll position
        const containers = document.querySelectorAll(".logits-display-container");
        for (const container of containers) {
            if (container !== target) {
                (container as HTMLElement).scrollLeft = scrollLeft;
            }
        }

        // Reset the flag after a short delay
        setTimeout(() => {
            isScrolling = false;
        }, 10);
    }
</script>

<main>
    <div class="container">
        <div class="main-layout">
            <!-- Left side: Static heatmap and controls -->
            <div class="left-panel">
                <div class="run-selector">
                    <label for="wandb-run-id">W&B Run ID</label>
                    <div class="input-group">
                        <input
                            type="text"
                            id="wandb-run-id"
                            list="run-options"
                            bind:value={wandbRunId}
                            disabled={loadingRun || isLoading}
                            placeholder="Select or enter run ID"
                        />
                        <datalist id="run-options">
                            {#each presetRuns as preset}
                                <option value={preset.id}>{preset.label}</option>
                            {/each}
                        </datalist>
                        <button
                            on:click={loadRun}
                            disabled={loadingRun || isLoading || !wandbRunId.trim()}
                        >
                            {loadingRun ? "Loading..." : "Load Run"}
                        </button>
                    </div>
                </div>

                <div class="prompt-section">
                    <textarea
                        id="prompt"
                        bind:value={prompt}
                        disabled={isLoading}
                        placeholder="Enter your prompt here..."
                        rows="2"
                        on:input={runPrompt}
                    ></textarea>
                </div>

                {#if result}
                    <div class="heatmap-container">
                        <!-- Layer labels column (fixed) -->
                        <div class="layer-labels">
                            <div class="layer-label-spacer"></div>
                            {#each result.layer_cis as layer}
                                <div class="layer-label">{layer.module}</div>
                            {/each}
                        </div>

                        <!-- Scrollable heatmap grid -->
                        <div class="heatmap-scroll-area">
                            <div class="heatmap-grid">
                                <!-- Heatmap cells -->
                                {#each result.layer_cis as layer, layerIdx}
                                    <div class="heatmap-row">
                                        {#each result.prompt_tokens as token, tokenIdx}
                                            <div
                                                class="heatmap-cell"
                                                style="background: {getColorFroml0(
                                                    layer.token_cis[tokenIdx].l0,
                                                    layer.module,
                                                    tokenIdx
                                                )}"
                                                title="L0={layer.token_cis[tokenIdx].l0}"
                                                on:click={() =>
                                                    openPopup(
                                                        token,
                                                        tokenIdx,
                                                        layer.module,
                                                        layerIdx,
                                                        layer.token_cis[tokenIdx]
                                                    )}
                                            ></div>
                                        {/each}
                                    </div>
                                {/each}

                                <!-- Token labels at the bottom -->
                                <div class="token-labels">
                                    {#each result.prompt_tokens as token}
                                        <div class="token-label">{token}</div>
                                    {/each}
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- Disabled Components Panel -->
                    <div class="disabled-components-panel">
                        <div class="disabled-header-row">
                            <h3>Disabled Components</h3>
                            <button
                                on:click={sendModification}
                                disabled={isLoading}
                                class="modify-button"
                            >
                                {isLoading ? "Sending..." : "Send Component Modifications"}
                            </button>
                        </div>
                        {#if Object.keys(runModification).some( (layer) => runModification[layer].some((tokenList) => tokenList.length > 0) )}
                            <div class="disabled-list">
                                {#each Object.entries(runModification) as [layerName, tokenArrays]}
                                    {#each tokenArrays as disabledComponents, tokenIdx}
                                        {#if disabledComponents.length > 0}
                                            <div class="disabled-group">
                                                <div class="disabled-header">
                                                    <strong>{result.prompt_tokens[tokenIdx]}</strong
                                                    >
                                                    in
                                                    <em>{layerName}</em>
                                                </div>
                                                <div class="disabled-items">
                                                    {#each disabledComponents as componentIdx}
                                                        <span
                                                            class="disabled-chip"
                                                            on:click={() =>
                                                                toggleComponentDisabled(
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
                {/if}
            </div>

            <!-- Right side: Scrollable predictions -->
            <div class="right-panel">
                <!-- Original Predictions (at top) -->
                {#if result && result.full_run_token_logits}
                    <div class="predictions-section">
                        <h2>Original Model Predictions</h2>
                        <div class="logits-display-container original" on:scroll={syncScroll}>
                            <div class="logits-display">
                                {#each result.full_run_token_logits as tokenPredictions, tokenIdx}
                                    <div class="token-predictions">
                                        <div class="token-header">
                                            <div class="token-name">
                                                "{result.prompt_tokens[tokenIdx]}"
                                            </div>
                                        </div>
                                        <div class="predictions-list">
                                            {#each tokenPredictions as prediction, predIdx}
                                                <div class="prediction-item">
                                                    <span class="prediction-token"
                                                        >"{prediction.token}"</span
                                                    >
                                                    <span class="prediction-prob"
                                                        >{prediction.probability.toFixed(3)}</span
                                                    >
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    </div>
                {/if}

                <!-- Original CI Masked Predictions (at top) -->
                {#if result && result.ci_masked_token_logits}
                    <div class="predictions-section">
                        <h2>Original <strong>CI Masked</strong> Model Predictions</h2>
                        <div class="logits-display-container original" on:scroll={syncScroll}>
                            <div class="logits-display">
                                {#each result.ci_masked_token_logits as tokenPredictions, tokenIdx}
                                    <div class="token-predictions">
                                        <div class="token-header">
                                            <div class="token-name">
                                                "{result.prompt_tokens[tokenIdx]}"
                                            </div>
                                        </div>
                                        <div class="predictions-list">
                                            {#each tokenPredictions as prediction, predIdx}
                                                <div class="prediction-item">
                                                    <span class="prediction-token"
                                                        >"{prediction.token}"</span
                                                    >
                                                    <span class="prediction-prob"
                                                        >{prediction.probability.toFixed(3)}</span
                                                    >
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    </div>
                {/if}

                <!-- Modification Results (stacked below) -->
                {#if result}
                    {#each modificationResults as modResult, modIdx}
                        <div class="modification-output-section">
                            <div class="logits-display-container" on:scroll={syncScroll}>
                                <div class="logits-display">
                                    {#each modResult.tokenLogits as tokenPredictions, tokenIdx}
                                        <div class="token-predictions">
                                            <div class="token-header">
                                                <div class="token-name">
                                                    "{result.prompt_tokens[tokenIdx]}"
                                                </div>
                                            </div>
                                            <div class="predictions-list">
                                                {#each tokenPredictions as prediction, predIdx}
                                                    <div class="prediction-item">
                                                        <span class="prediction-token"
                                                            >"{prediction.token}"</span
                                                        >
                                                        <span class="prediction-prob"
                                                            >{prediction.probability.toFixed(
                                                                3
                                                            )}</span
                                                        >
                                                    </div>
                                                {/each}
                                            </div>
                                        </div>
                                    {/each}
                                </div>
                            </div>

                            <div class="modification-summary">
                                <h3>Applied Modifications:</h3>
                                <div class="applied-modifications">
                                    {#each Object.entries(modResult.applied_mask) as [layerName, tokenArrays]}
                                        {#each tokenArrays as disabledComponents, tokenIdx}
                                            {#if disabledComponents.length > 0}
                                                <div class="applied-mod-item">
                                                    <strong>{result.prompt_tokens[tokenIdx]}</strong
                                                    >
                                                    in
                                                    <em>{layerName}</em>: disabled components {disabledComponents.join(
                                                        ", "
                                                    )}
                                                </div>
                                            {/if}
                                        {/each}
                                    {/each}
                                </div>
                            </div>
                        </div>
                    {/each}
                {/if}
            </div>
        </div>

        <!-- Popup Modal -->
        {#if popupData}
            <div class="popup-overlay" on:click={closePopup}>
                <div class="popup-modal" on:click|stopPropagation>
                    <div class="popup-content">
                        <div class="popup-info">
                            <p>
                                <strong>Token:</strong> "{popupData.token}" (position {popupData.tokenIdx})
                            </p>
                            <p><strong>Layer:</strong> {popupData.layer}</p>
                            <p><strong>L0 (Non-zero components):</strong> {popupData.tokenCi.l0}</p>
                            <p>
                                <strong>Vector Length:</strong>
                                {popupData.tokenCi.component_cis.length}
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
                                {#each popupData.tokenCi.component_cis as value, idx}
                                    <div
                                        class="vector-item"
                                        style="--item-bg-color: {getColorFromCI(value / 3)}"
                                        class:disabled={isComponentDisabled(
                                            popupData.layer,
                                            popupData.tokenIdx,
                                            popupData.tokenCi.indices[idx]
                                        )}
                                        on:click={() => {
                                            if (popupData) {
                                                toggleComponentDisabled(
                                                    popupData.layer,
                                                    popupData.tokenIdx,
                                                    popupData.tokenCi.indices[idx]
                                                );
                                            }
                                        }}
                                    >
                                        <span class="component-idx"
                                            >{popupData.tokenCi.indices[idx]}:</span
                                        >
                                        <span class="component-value">{value.toFixed(4)}</span>
                                        <!-- <span class="disable-indicator">
											{isComponentDisabled(popupData.layer, popupData.tokenIdx, idx) ? '🚫' : '✓'}
										</span> -->
                                    </div>
                                {/each}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        {/if}
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
        gap: 0.5rem;
    }

    .main-layout {
        display: flex;
        gap: 1rem;
        min-height: 80vh;
    }

    .left-panel {
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

    .run-selector {
        margin-bottom: 1rem;
    }

    .run-selector label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #333;
        font-size: 0.9rem;
    }

    .input-group {
        display: flex;
        gap: 0.5rem;
    }

    .input-group input[type="text"] {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1rem;
    }

    .input-group input[type="text"]:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }

    .input-group button {
        padding: 0.5rem 1rem;
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 1rem;
        cursor: pointer;
        white-space: nowrap;
    }

    .input-group button:hover:not(:disabled) {
        background-color: #357abd;
    }

    .input-group button:disabled {
        background-color: #ccc;
        cursor: not-allowed;
    }

    .prompt-section {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    label {
        font-weight: bold;
        color: #333;
    }

    textarea {
        padding: 0.5rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 1rem;
        resize: vertical;
    }

    button {
        padding: 0.75rem 1.5rem;
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
    }

    button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }

    button:hover:not(:disabled) {
        background-color: #45a049;
    }

    .modify-button {
        background-color: #ff6b35;
    }

    .modify-button:hover:not(:disabled) {
        background-color: #e55a2b;
    }

    .heatmap-container {
        flex: 1;
        display: flex;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1rem;
        background-color: #fafafa;
    }

    .disabled-components-panel {
        flex: 0 0 300px;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: #f8f9fa;
        max-height: 500px;
        overflow-y: auto;
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

    .layer-labels {
        display: flex;
        flex-direction: column;
        margin-right: 0.5rem;
        flex-shrink: 0;
    }

    .layer-label-spacer {
        height: 40px; /* Space for token labels at bottom */
        order: 999; /* Move to bottom to align with token labels */
    }

    .layer-label {
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 0.5rem;
        font-size: 0.9rem;
        font-weight: bold;
        color: #555;
        min-width: 100px;
        margin-bottom: 2px;
    }

    .heatmap-scroll-area {
        flex: 1;
        overflow-x: auto;
    }

    .heatmap-grid {
        display: flex;
        flex-direction: column;
        min-width: fit-content;
    }

    .token-labels {
        display: flex;
        height: 40px;
        margin-bottom: 2px;
    }

    .token-label {
        width: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: bold;
        color: #333;
        text-align: center;
        padding: 0 2px;
        word-break: break-all;
        border-right: 1px solid #eee;
    }

    .heatmap-row {
        display: flex;
        margin-bottom: 2px;
    }

    .heatmap-cell {
        width: 50px;
        height: 20px;
        border: 1px solid #fff;
        cursor: pointer;
        transition: transform 0.1s ease;
    }

    .heatmap-cell:hover {
        border: 2px solid #241d8c;
        z-index: 10;
        position: relative;
    }

    h2 {
        margin-bottom: 1rem;
        color: #333;
    }

    /* Popup Modal Styles */
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
        /* border: 1px solid #ff6b6b; */
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

    /* Predictions Sections */
    .predictions-section {
        padding: 1rem;
        border: 2px solid #4caf50;
        border-radius: 8px;
        background-color: #f8fff8;
        margin-bottom: 1rem;
    }

    .predictions-section h2 {
        margin: 0 0 0.5rem 0;
        color: #2e7d2e;
    }

    .logits-display-container {
        overflow-x: auto;
        margin-bottom: 0;
        border: 1px solid #ddd;
        border-radius: 6px;
        background: white;
        /* Hide scrollbar on firefox: */
        scrollbar-width: none;
    }

    /* Hide scrollbar on webkit browsers: */
    .logits-display-container::-webkit-scrollbar {
        display: none;
    }

    .logits-display-container.original {
        border-color: #4caf50;
    }

    .logits-display {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 0;
        min-width: fit-content;
        padding: 0.5rem;
        width: max-content;
        overflow: visible;
    }

    .token-name {
        font-size: 0.8rem;
        color: #333;
        font-family: monospace;
        margin-top: 0.2rem;
        word-break: break-all;
    }

    .predictions-list {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
    }

    .prediction-item {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        border-radius: 2px;
        font-size: 0.7rem;
    }

    .prediction-token {
        font-family: monospace;
        font-size: 0.7rem;
        color: #333;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        text-align: left;
        flex: 1;
        margin-right: 0.2rem;
        padding: 0.1rem 0.3rem;
        border-radius: 2px;
    }

    .prediction-prob {
        font-family: monospace;
        font-size: 0.65rem;
        text-align: right;
        flex-shrink: 0;
    }

    /* Modification Output Styles */
    .modification-output-section {
        margin-top: 1rem;
        padding: 0.75rem;
        border: 2px solid #ff6b35;
        border-radius: 8px;
        background-color: #fff8f5;
    }

    .token-predictions {
        background: white;
        border-radius: 6px;
        border: 1px solid #ddd;
        width: 140px;
        padding: 0.25rem;
        margin-right: 0.25rem;
    }

    .token-header {
        color: #333;
        text-align: center;
    }

    .modification-summary {
        background: white;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #ddd;
        margin-top: 0.5rem;
    }

    .modification-summary h3 {
        margin: 0 0 0.25rem 0;
        color: #333;
    }

    .applied-modifications {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .applied-mod-item {
        padding: 0.25rem;
        background-color: #f8f9fa;
        border-radius: 4px;
        border-left: 3px solid #ff6b35;
        font-size: 0.85rem;
        color: #555;
    }
</style>
