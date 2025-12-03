<script lang="ts">
    import * as api from "../lib/localAttributionsApi";
    import type {
        PromptPreview,
        PromptData,
        ActivationContextsSummary,
        PinnedNode,
        TokenizeResult,
    } from "../lib/localAttributionsTypes";
    import LocalAttributionsGraph from "./LocalAttributionsGraph.svelte";

    // Server state
    let serverStatus = $state<api.ServerStatus | null>(null);
    let serverError = $state<string | null>(null);

    // Prompt state
    let prompts = $state<PromptPreview[]>([]);
    let currentPromptId = $state<number | null>(null);
    let promptData = $state<PromptData | null>(null);
    let loadingPrompt = $state(false);
    let promptError = $state<string | null>(null);

    // Custom prompt state
    let customPromptText = $state("");
    let tokenizedPreview = $state<TokenizeResult | null>(null);
    let tokenizeLoading = $state(false);
    let customPromptError = $state<string | null>(null);

    // Prompt generation state
    let generatingPrompts = $state(false);
    let generateProgress = $state(0);
    let generateCount = $state(0);
    let generateError = $state<string | null>(null);

    // Activation contexts (for filtering and component details)
    let activationContextsSummary = $state<ActivationContextsSummary | null>(null);
    let activationContextsMissing = $state(false);

    // Controls
    let maxMeanCI = $state(1.0);
    let topK = $state(800);
    let normalizeEdges = $state(true);
    let nodeLayout = $state<"importance" | "shuffled" | "jittered">("importance");

    // Optimization controls
    let useOptimized = $state(false);
    let impMinCoeff = $state(0.1);
    let ceLossCoeff = $state(1.0);
    let optimSteps = $state(500);
    let optimPnorm = $state(0.3);

    // Pinned nodes (for search)
    let pinnedNodes = $state<PinnedNode[]>([]);

    // Search state
    let searchResults = $state<PromptPreview[]>([]);
    let searchLoading = $state(false);
    let searchError = $state<string | null>(null);

    // Derived: is a run loaded?
    const loadedRun = $derived(serverStatus?.loaded_run ?? null);

    // Load server status once on mount (App.svelte handles polling)
    $effect(() => {
        loadServerStatus();
    });

    // When loaded run changes, refresh data
    let previousRunId: number | null = null;
    $effect(() => {
        const currentRunId = loadedRun?.id ?? null;
        if (currentRunId !== null && currentRunId !== previousRunId) {
            console.log(`[LocalAttr] run changed: ${previousRunId} -> ${currentRunId}, loading data...`);
            previousRunId = currentRunId;
            loadPromptsList();
            loadActivationContextsSummary();
        } else if (currentRunId === null && previousRunId !== null) {
            // Run was unloaded
            console.log(`[LocalAttr] run unloaded (was ${previousRunId})`);
            previousRunId = null;
            prompts = [];
            promptData = null;
            currentPromptId = null;
            activationContextsSummary = null;
            activationContextsMissing = false;
        }
    });

    async function loadServerStatus() {
        const t0 = performance.now();
        try {
            serverStatus = await api.getStatus();
            serverError = null;
            console.log(`[LocalAttr] loadServerStatus: ${(performance.now() - t0).toFixed(0)}ms, run=${serverStatus?.loaded_run?.id ?? "none"}`);
        } catch (e) {
            serverError = e instanceof Error ? e.message : "Failed to connect to server";
            console.error(`[LocalAttr] loadServerStatus FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        }
    }

    async function loadPromptsList() {
        const t0 = performance.now();
        console.log("[LocalAttr] loadPromptsList: starting...");
        try {
            prompts = await api.listPrompts();
            console.log(`[LocalAttr] loadPromptsList: got ${prompts.length} prompts in ${(performance.now() - t0).toFixed(0)}ms`);
            // Don't auto-load first prompt - let user click "Compute Graph"
            if (prompts.length > 0 && currentPromptId === null) {
                currentPromptId = prompts[0].id;
            }
        } catch (e) {
            console.error(`[LocalAttr] loadPromptsList FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        }
    }

    async function loadActivationContextsSummary() {
        const t0 = performance.now();
        console.log("[LocalAttr] loadActivationContextsSummary: starting...");
        try {
            activationContextsSummary = await api.getActivationContextsSummary();
            activationContextsMissing = false;
            const layerCount = Object.keys(activationContextsSummary).length;
            console.log(`[LocalAttr] loadActivationContextsSummary: got ${layerCount} layers in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            // Check if it's a "missing" error (404 with missing flag)
            if (e instanceof Error && e.message.includes("404")) {
                activationContextsMissing = true;
                activationContextsSummary = null;
                console.log(`[LocalAttr] loadActivationContextsSummary: not found (${(performance.now() - t0).toFixed(0)}ms)`);
            } else {
                console.error(`[LocalAttr] loadActivationContextsSummary FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
            }
        }
    }

    async function loadPrompt(promptId: number) {
        if (loadingPrompt) {
            console.log(`[LocalAttr] loadPrompt(${promptId}): skipped, already loading`);
            return;
        }

        loadingPrompt = true;
        promptError = null;
        currentPromptId = promptId;

        const t0 = performance.now();
        const mode = useOptimized ? "optimized" : "standard";
        console.log(`[LocalAttr] loadPrompt(${promptId}): starting (${mode})...`);

        try {
            if (useOptimized) {
                promptData = await api.getPromptOptimized(promptId, {
                    maxMeanCI,
                    normalize: normalizeEdges,
                    impMinCoeff,
                    ceLossCoeff,
                    steps: optimSteps,
                    pnorm: optimPnorm,
                });
            } else {
                promptData = await api.getPrompt(promptId, {
                    maxMeanCI,
                    normalize: normalizeEdges,
                });
            }
            const edgeCount = promptData?.edges?.length ?? 0;
            console.log(`[LocalAttr] loadPrompt(${promptId}): got ${edgeCount} edges in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            promptError = e instanceof Error ? e.message : "Failed to load prompt";
            console.error(`[LocalAttr] loadPrompt(${promptId}) FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            loadingPrompt = false;
        }
    }

    function handlePromptSelect(event: Event) {
        const target = event.target as HTMLSelectElement;
        const promptId = parseInt(target.value);
        console.log(`[LocalAttr] handlePromptSelect: selected ${promptId}`);
        if (!isNaN(promptId)) {
            currentPromptId = promptId;
            // Don't auto-load - let user click "Compute Graph"
        }
    }

    async function reloadWithFilters() {
        console.log(`[LocalAttr] reloadWithFilters: currentPromptId=${currentPromptId}`);
        if (currentPromptId !== null) {
            await loadPrompt(currentPromptId);
        }
    }

    async function searchPrompts() {
        if (pinnedNodes.length === 0) return;

        searchLoading = true;
        searchError = null;

        const t0 = performance.now();
        const components = pinnedNodes.map((p) => `${p.layer}:${p.cIdx}`);
        console.log(`[LocalAttr] searchPrompts: searching for ${components.length} components...`);

        try {
            const result = await api.searchPrompts(components, "all");
            searchResults = result.results;
            console.log(`[LocalAttr] searchPrompts: found ${searchResults.length} results in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            searchError = e instanceof Error ? e.message : "Search failed";
            console.error(`[LocalAttr] searchPrompts FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            searchLoading = false;
        }
    }

    function handlePinnedNodesChange(nodes: PinnedNode[]) {
        pinnedNodes = nodes;
    }

    // Custom prompt functions
    async function tokenizeCustomPrompt() {
        if (!customPromptText.trim()) return;

        tokenizeLoading = true;
        customPromptError = null;
        tokenizedPreview = null;

        const t0 = performance.now();
        console.log(`[LocalAttr] tokenizeCustomPrompt: tokenizing ${customPromptText.length} chars...`);

        try {
            tokenizedPreview = await api.tokenizeText(customPromptText);
            console.log(`[LocalAttr] tokenizeCustomPrompt: got ${tokenizedPreview.tokens.length} tokens in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            customPromptError = e instanceof Error ? e.message : "Failed to tokenize";
            console.error(`[LocalAttr] tokenizeCustomPrompt FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            tokenizeLoading = false;
        }
    }

    async function computeCustomPromptGraph() {
        if (!tokenizedPreview) return;

        loadingPrompt = true;
        promptError = null;
        currentPromptId = null; // Clear since this is a custom prompt

        const t0 = performance.now();
        console.log(`[LocalAttr] computeCustomPromptGraph: computing for ${tokenizedPreview.tokens.length} tokens...`);

        try {
            promptData = await api.computeCustomPrompt({
                tokenIds: tokenizedPreview.token_ids,
                normalize: normalizeEdges,
            });
            const edgeCount = promptData?.edges?.length ?? 0;
            console.log(`[LocalAttr] computeCustomPromptGraph: got ${edgeCount} edges in ${(performance.now() - t0).toFixed(0)}ms`);
            // Clear tokenization preview after successful compute
            tokenizedPreview = null;
            customPromptText = "";
        } catch (e) {
            promptError = e instanceof Error ? e.message : "Failed to compute graph";
            console.error(`[LocalAttr] computeCustomPromptGraph FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            loadingPrompt = false;
        }
    }

    function cancelTokenization() {
        tokenizedPreview = null;
    }

    // Prompt generation
    async function handleGeneratePrompts(nPrompts: number = 100) {
        if (generatingPrompts) {
            console.log("[LocalAttr] handleGeneratePrompts: skipped, already generating");
            return;
        }

        generatingPrompts = true;
        generateProgress = 0;
        generateCount = 0;
        generateError = null;

        const t0 = performance.now();
        console.log(`[LocalAttr] handleGeneratePrompts: starting generation of ${nPrompts} prompts...`);

        try {
            await api.generatePrompts(
                { nPrompts },
                (progress, count) => {
                    generateProgress = progress;
                    generateCount = count;
                    if (count % 10 === 0) {
                        console.log(`[LocalAttr] handleGeneratePrompts: progress ${count}/${nPrompts}`);
                    }
                },
            );
            console.log(`[LocalAttr] handleGeneratePrompts: completed in ${(performance.now() - t0).toFixed(0)}ms`);
            // Refresh prompts list after generation
            await loadPromptsList();
        } catch (e) {
            generateError = e instanceof Error ? e.message : "Failed to generate prompts";
            console.error(`[LocalAttr] handleGeneratePrompts FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            generatingPrompts = false;
        }
    }
</script>

<div class="local-attributions-tab">
    {#if !loadedRun}
        <!-- No run loaded state -->
        <div class="no-run-message">
            <p>No run loaded. Select a run from the sidebar to view local attributions.</p>
            {#if serverError}
                <p class="server-error">{serverError}</p>
            {/if}
        </div>
    {:else}
        <!-- Sidebar -->
        <div class="sidebar">
            {#if pinnedNodes.length > 0}
            <div class="sidebar-section">
                <h3>Search Prompts</h3>
                <button class="search-btn" onclick={searchPrompts} disabled={searchLoading}>
                    {searchLoading ? "Searching..." : `Find prompts with ${pinnedNodes.length} pinned`}
                </button>
                {#if searchError}
                    <div class="search-error">{searchError}</div>
                {/if}
                {#if searchResults.length > 0}
                    <div class="search-info">Found {searchResults.length} prompts</div>
                    <div class="search-results">
                        {#each searchResults as p}
                            <button
                                class="search-result"
                                class:active={p.id === currentPromptId}
                                onclick={() => loadPrompt(p.id)}
                            >
                                <div class="search-result-id">#{p.id}</div>
                                <div class="search-result-preview">{p.preview}</div>
                            </button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/if}
        </div>

        <!-- Main content -->
        <div class="main-content">
        {#if activationContextsMissing}
            <div class="warning-banner">
                <strong>⚠️ Activation contexts not generated.</strong>
                Component hover info will be unavailable. Generate via the API or batch script.
            </div>
        {/if}

        {#if promptError}
            <div class="error-banner">
                {promptError}
                <button onclick={() => currentPromptId && loadPrompt(currentPromptId)}>Retry</button>
            </div>
        {/if}

        {#if generateError}
            <div class="error-banner">
                {generateError}
                <button onclick={() => (generateError = null)}>Dismiss</button>
            </div>
        {/if}

        <!-- Custom prompt input section -->
        <div class="custom-prompt-section">
            <div class="custom-prompt-input-row">
                <input
                    type="text"
                    bind:value={customPromptText}
                    placeholder="Enter custom text to analyze..."
                    onkeydown={(e) => e.key === "Enter" && tokenizeCustomPrompt()}
                    class="custom-prompt-input"
                    disabled={tokenizeLoading}
                />
                <button onclick={tokenizeCustomPrompt} disabled={!customPromptText.trim() || tokenizeLoading} class="tokenize-btn">
                    {#if tokenizeLoading}
                        <span class="loading-spinner small"></span>
                    {:else}
                        Tokenize
                    {/if}
                </button>
            </div>

            {#if tokenizedPreview}
                <div class="tokenization-preview">
                    <div class="tokenization-header">
                        <span><strong>{tokenizedPreview.tokens.length}</strong> tokens:</span>
                        <button onclick={computeCustomPromptGraph} disabled={loadingPrompt} class="compute-btn">
                            {#if loadingPrompt}
                                <span class="loading-spinner small"></span> Computing...
                            {:else}
                                Compute Graph
                            {/if}
                        </button>
                        <button onclick={cancelTokenization} class="cancel-btn">Cancel</button>
                    </div>
                    <div class="token-preview-list">
                        {#each tokenizedPreview.tokens as tok}
                            <span class="token-preview-item">{tok}</span>
                        {/each}
                    </div>
                </div>
            {/if}

            {#if customPromptError}
                <div class="custom-prompt-error">{customPromptError}</div>
            {/if}
        </div>

        <div class="controls">
            <div class="prompt-controls">
                {#if prompts.length === 0 && !generatingPrompts}
                    <button class="generate-btn" onclick={() => handleGeneratePrompts(100)}>
                        Generate 100 prompts
                    </button>
                {:else if generatingPrompts}
                    <div class="generate-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {generateProgress * 100}%"></div>
                        </div>
                        <span class="progress-text">{generateCount} prompts...</span>
                    </div>
                {:else}
                    <select onchange={handlePromptSelect} disabled={loadingPrompt}>
                        <option value="">Select a prompt...</option>
                        {#each prompts as p}
                            <option value={p.id} selected={p.id === currentPromptId}>
                                {p.id}: {p.preview}
                            </option>
                        {/each}
                    </select>
                    <button
                        class="compute-graph-btn"
                        onclick={() => currentPromptId && loadPrompt(currentPromptId)}
                        disabled={loadingPrompt || currentPromptId === null}
                    >
                        {#if loadingPrompt}
                            Computing...
                        {:else}
                            Compute Graph
                        {/if}
                    </button>
                {/if}
            </div>

            <label>
                Max mean CI:
                <input type="number" bind:value={maxMeanCI} min={0} max={1} step={0.01} onchange={reloadWithFilters} />
            </label>

            <label>
                Top K edges:
                <input type="number" bind:value={topK} min={10} max={10000} step={100} />
            </label>

            <label>
                Layout:
                <select bind:value={nodeLayout}>
                    <option value="importance">By importance</option>
                    <option value="shuffled">Shuffled</option>
                    <option value="jittered">Jittered</option>
                </select>
            </label>

            <label class="checkbox-label">
                <input type="checkbox" bind:checked={normalizeEdges} onchange={reloadWithFilters} />
                Normalize edges
            </label>

            <label class="checkbox-label">
                <input type="checkbox" bind:checked={useOptimized} />
                Optimize CI
            </label>
        </div>

        {#if useOptimized}
            <div class="optim-controls">
                <label>
                    imp_min:
                    <input type="number" bind:value={impMinCoeff} min={0.001} max={10} step={0.01} />
                </label>
                <label>
                    ce:
                    <input type="number" bind:value={ceLossCoeff} min={0.001} max={10} step={0.1} />
                </label>
                <label>
                    steps:
                    <input type="number" bind:value={optimSteps} min={10} max={5000} step={100} />
                </label>
                <label>
                    pnorm:
                    <input type="number" bind:value={optimPnorm} min={0.1} max={1} step={0.1} />
                </label>
                <button class="run-btn" onclick={reloadWithFilters} disabled={loadingPrompt}>Run</button>
            </div>

            {#if promptData?.optimization}
                <div class="optim-results">
                    <span>
                        <strong>Target:</strong> "{promptData.optimization.label_str}"
                        @ {(promptData.optimization.label_prob * 100).toFixed(1)}%
                    </span>
                    <span>
                        <strong>L0:</strong>
                        {promptData.optimization.l0_total.toFixed(0)} active
                    </span>
                </div>
            {/if}
        {/if}

        <div class="graph-area" class:loading={loadingPrompt}>
            {#if loadingPrompt}
                <div class="loading-overlay">
                    <div class="loading-spinner"></div>
                    <span>Loading prompt...</span>
                </div>
            {/if}

            {#if promptData}
                <LocalAttributionsGraph
                    data={promptData}
                    {topK}
                    {nodeLayout}
                    {activationContextsSummary}
                    {pinnedNodes}
                    onPinnedNodesChange={handlePinnedNodesChange}
                />
            {:else if !loadingPrompt}
                <div class="empty-state">
                    <p>Select a prompt to view its attribution graph</p>
                    <p class="empty-state-hint">{prompts.length} prompts available</p>
                </div>
            {/if}
        </div>
    </div>
    {/if}
</div>

<style>
    .local-attributions-tab {
        display: flex;
        height: 100%;
        min-height: 0;
    }

    .sidebar {
        width: 280px;
        background: #fff;
        border-right: 1px solid #e0e0e0;
        display: flex;
        flex-direction: column;
        flex-shrink: 0;
        overflow-y: auto;
    }

    .sidebar-section {
        padding: 1rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .sidebar-section h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: #333;
    }

    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 0;
        padding: 1rem;
        gap: 0.75rem;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 0.75rem 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        flex-wrap: wrap;
    }

    .controls label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
    }

    .controls input[type="number"] {
        width: 80px;
        padding: 0.25rem 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .controls select {
        padding: 0.25rem 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        min-width: 150px;
    }

    .prompt-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .compute-graph-btn {
        padding: 0.4rem 0.8rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        white-space: nowrap;
    }

    .compute-graph-btn:hover:not(:disabled) {
        background: #1976d2;
    }

    .compute-graph-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .generate-btn {
        padding: 0.4rem 0.8rem;
        background: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .generate-btn:hover {
        background: #388e3c;
    }

    .generate-progress {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        min-width: 180px;
    }

    .progress-bar {
        flex: 1;
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: #4caf50;
        transition: width 0.1s ease;
    }

    .progress-text {
        font-size: 0.8rem;
        color: #666;
        white-space: nowrap;
    }

    .checkbox-label {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .optim-controls {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 1rem;
        background: #e8f5e9;
        border-radius: 4px;
    }

    .optim-controls label {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.85rem;
    }

    .optim-controls input {
        width: 70px;
        padding: 0.25rem;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .run-btn {
        padding: 0.4rem 0.8rem;
        background: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }

    .run-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .optim-results {
        display: flex;
        gap: 1rem;
        padding: 0.5rem 0.75rem;
        background: #fff3e0;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .graph-area {
        flex: 1;
        position: relative;
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        min-height: 400px;
        overflow: hidden;
    }

    .graph-area.loading {
        opacity: 0.7;
    }

    .loading-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.8);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        z-index: 100;
    }

    .loading-spinner {
        width: 24px;
        height: 24px;
        border: 3px solid #e0e0e0;
        border-top-color: #2196f3;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #666;
        font-size: 0.95rem;
        text-align: center;
    }

    .empty-state p {
        margin: 0.25rem 0;
    }

    .empty-state-hint {
        font-size: 0.85rem;
        color: #999;
    }

    /* No run loaded message */
    .no-run-message {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: #666;
        text-align: center;
        padding: 2rem;
    }

    .no-run-message .server-error {
        color: #c62828;
        margin-top: 0.5rem;
    }

    /* Custom prompt section */
    .custom-prompt-section {
        padding: 0.75rem 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .custom-prompt-input-row {
        display: flex;
        gap: 0.5rem;
    }

    .custom-prompt-input {
        flex: 1;
        padding: 0.5rem 0.75rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    .custom-prompt-input:focus {
        outline: none;
        border-color: #2196f3;
    }

    .tokenize-btn,
    .compute-btn {
        padding: 0.5rem 1rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .tokenize-btn:hover:not(:disabled),
    .compute-btn:hover:not(:disabled) {
        background: #1976d2;
    }

    .tokenize-btn:disabled,
    .compute-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .compute-btn {
        background: #4caf50;
    }

    .compute-btn:hover:not(:disabled) {
        background: #388e3c;
    }

    .cancel-btn {
        padding: 0.5rem 1rem;
        background: transparent;
        color: #666;
        border: 1px solid #ddd;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .cancel-btn:hover {
        background: #f5f5f5;
    }

    .tokenization-preview {
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: #f5f5f5;
        border-radius: 4px;
    }

    .tokenization-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }

    .token-preview-list {
        display: flex;
        flex-wrap: wrap;
        gap: 2px;
    }

    .token-preview-item {
        padding: 2px 6px;
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 3px;
        font-family: monospace;
        font-size: 0.85rem;
    }

    .custom-prompt-error {
        margin-top: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: #ffebee;
        border: 1px solid #f44336;
        border-radius: 4px;
        color: #c62828;
        font-size: 0.85rem;
    }

    .loading-spinner.small {
        width: 16px;
        height: 16px;
        border-width: 2px;
    }

    .warning-banner {
        padding: 0.75rem 1rem;
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 4px;
        color: #e65100;
        font-size: 0.9rem;
    }

    .error-banner {
        padding: 0.75rem 1rem;
        background: #ffebee;
        border: 1px solid #f44336;
        border-radius: 4px;
        color: #c62828;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .error-banner button {
        margin-left: auto;
        padding: 0.25rem 0.5rem;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .search-btn {
        width: 100%;
        padding: 0.5rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .search-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
    }

    .search-error {
        padding: 0.5rem;
        background: #ffebee;
        color: #c62828;
        font-size: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }

    .search-info {
        padding: 0.5rem;
        background: #e3f2fd;
        color: #1565c0;
        font-size: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }

    .search-results {
        max-height: 300px;
        overflow-y: auto;
        margin-top: 0.5rem;
    }

    .search-result {
        width: 100%;
        padding: 0.5rem;
        background: #fafafa;
        border: 1px solid #eee;
        border-radius: 4px;
        cursor: pointer;
        text-align: left;
        margin-bottom: 0.25rem;
    }

    .search-result:hover {
        background: #f5f5f5;
    }

    .search-result.active {
        background: #e3f2fd;
        border-color: #2196f3;
    }

    .search-result-id {
        font-size: 0.75rem;
        color: #666;
    }

    .search-result-preview {
        font-family: monospace;
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
