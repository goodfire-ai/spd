<script lang="ts">
    import * as attrApi from "../lib/localAttributionsApi";
    import * as mainApi from "../lib/api";
    import type {
        PromptPreview,
        PromptData,
        ActivationContextsSummary,
        PinnedNode,
        TokenizeResult,
    } from "../lib/localAttributionsTypes";
    import LocalAttributionsGraph from "./LocalAttributionsGraph.svelte";

    // Server state
    let loadedRun = $state<mainApi.LoadedRun | null>(null);
    let serverError = $state<string | null>(null);

    // Prompt state
    let prompts = $state<PromptPreview[]>([]);
    let currentPromptId = $state<number | null>(null);
    let loadingPrompt = $state(false);
    let loadingMode = $state<"standard" | "optimized">("standard");
    let loadingProgress = $state<{ current: number; total: number; stage: string } | null>(null);
    let promptError = $state<string | null>(null);

    // Custom prompt state
    let customPromptText = $state("");
    let tokenizedPreview = $state<TokenizeResult | null>(null);
    let tokenizeLoading = $state(false);
    let customPromptError = $state<string | null>(null);

    // Cached graphs state (multiple graphs per prompt for comparison)
    type CachedGraph = { id: string; label: string; data: PromptData };
    let cachedGraphs = $state<CachedGraph[]>([]);
    let activeGraphId = $state<string | null>(null);

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

    // Search/filter state
    let filterByPinned = $state(false);
    let filteredPrompts = $state<PromptPreview[]>([]);
    let filterLoading = $state(false);

    // Derived: prompts to display (filtered or all)
    const displayedPrompts = $derived(filterByPinned ? filteredPrompts : prompts);

    // Derived: staged tokens (from selected prompt or custom tokenization)
    const stagedTokens = $derived.by(() => {
        if (tokenizedPreview) {
            return { tokens: tokenizedPreview.tokens, tokenIds: tokenizedPreview.token_ids, isCustom: true };
        }
        if (currentPromptId !== null) {
            const prompt = prompts.find((p) => p.id === currentPromptId);
            if (prompt) {
                return { tokens: prompt.tokens, tokenIds: null, isCustom: false };
            }
        }
        return null;
    });

    // Derived: active graph data
    const activeGraph = $derived(cachedGraphs.find((g) => g.id === activeGraphId) ?? null);

    // Load server status once on mount
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
            // Clear cached graphs when run changes
            cachedGraphs = [];
            activeGraphId = null;
        } else if (currentRunId === null && previousRunId !== null) {
            console.log(`[LocalAttr] run unloaded (was ${previousRunId})`);
            previousRunId = null;
            prompts = [];
            currentPromptId = null;
            cachedGraphs = [];
            activeGraphId = null;
            activationContextsSummary = null;
            activationContextsMissing = false;
        }
    });

    // Clear cached graphs when prompt changes
    let previousPromptId: number | null = null;
    $effect(() => {
        if (currentPromptId !== previousPromptId) {
            previousPromptId = currentPromptId;
            cachedGraphs = [];
            activeGraphId = null;
        }
    });

    async function loadServerStatus() {
        const t0 = performance.now();
        try {
            loadedRun = await mainApi.getStatus();
            serverError = null;
            console.log(`[LocalAttr] loadServerStatus: ${(performance.now() - t0).toFixed(0)}ms, run=${loadedRun?.id ?? "none"}`);
        } catch (e) {
            serverError = e instanceof Error ? e.message : "Failed to connect to server";
            console.error(`[LocalAttr] loadServerStatus FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        }
    }

    async function loadPromptsList() {
        const t0 = performance.now();
        console.log("[LocalAttr] loadPromptsList: starting...");
        try {
            prompts = await attrApi.listPrompts();
            console.log(`[LocalAttr] loadPromptsList: got ${prompts.length} prompts in ${(performance.now() - t0).toFixed(0)}ms`);
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
            activationContextsSummary = await attrApi.getActivationContextsSummary();
            activationContextsMissing = false;
            const layerCount = Object.keys(activationContextsSummary).length;
            console.log(`[LocalAttr] loadActivationContextsSummary: got ${layerCount} layers in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            const status = (e as { status?: number }).status;
            if (status === 404) {
                activationContextsMissing = true;
                activationContextsSummary = null;
                console.log(`[LocalAttr] loadActivationContextsSummary: not found (${(performance.now() - t0).toFixed(0)}ms)`);
            } else {
                console.error(`[LocalAttr] loadActivationContextsSummary FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
            }
        }
    }

    async function computeGraph() {
        if (loadingPrompt) return;
        if (!stagedTokens) return;

        loadingPrompt = true;
        loadingMode = useOptimized ? "optimized" : "standard";
        loadingProgress = null;
        promptError = null;

        const t0 = performance.now();
        const isOptimized = useOptimized;
        console.log(`[LocalAttr] computeGraph: starting (${loadingMode})...`);

        try {
            let data: PromptData;

            if (stagedTokens.isCustom && stagedTokens.tokenIds) {
                // Custom prompt
                data = await attrApi.computeCustomPrompt({
                    tokenIds: stagedTokens.tokenIds,
                    normalize: normalizeEdges,
                });
            } else if (currentPromptId !== null) {
                // Existing prompt
                if (isOptimized) {
                    data = await attrApi.getPromptOptimized(currentPromptId, {
                        maxMeanCI,
                        normalize: normalizeEdges,
                        impMinCoeff,
                        ceLossCoeff,
                        steps: optimSteps,
                        pnorm: optimPnorm,
                    });
                } else {
                    data = await attrApi.getPromptStreaming(
                        currentPromptId,
                        { maxMeanCI, normalize: normalizeEdges },
                        (progress) => {
                            loadingProgress = progress;
                        },
                    );
                }
            } else {
                throw new Error("No prompt selected");
            }

            // Generate unique ID and label for this graph
            const graphId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
            const label = isOptimized ? `Optimized (${optimSteps} steps)` : "Standard";

            // Add to cache
            cachedGraphs = [...cachedGraphs, { id: graphId, label, data }];
            activeGraphId = graphId;

            // Clear custom prompt state after successful compute
            if (stagedTokens.isCustom) {
                tokenizedPreview = null;
                customPromptText = "";
            }

            const edgeCount = data?.edges?.length ?? 0;
            console.log(`[LocalAttr] computeGraph: got ${edgeCount} edges in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            promptError = e instanceof Error ? e.message : "Failed to compute graph";
            console.error(`[LocalAttr] computeGraph FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            loadingPrompt = false;
            loadingProgress = null;
        }
    }

    function closeGraphTab(graphId: string) {
        cachedGraphs = cachedGraphs.filter((g) => g.id !== graphId);
        if (activeGraphId === graphId) {
            activeGraphId = cachedGraphs.length > 0 ? cachedGraphs[cachedGraphs.length - 1].id : null;
        }
    }

    async function filterPromptsByPinned() {
        if (pinnedNodes.length === 0) {
            filteredPrompts = [];
            return;
        }

        filterLoading = true;
        const t0 = performance.now();
        const components = pinnedNodes.map((p) => `${p.layer}:${p.cIdx}`);
        console.log(`[LocalAttr] filterPromptsByPinned: filtering for ${components.length} components...`);

        try {
            const result = await attrApi.searchPrompts(components, "all");
            filteredPrompts = result.results;
            console.log(`[LocalAttr] filterPromptsByPinned: found ${filteredPrompts.length} results in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            console.error(`[LocalAttr] filterPromptsByPinned FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
            filteredPrompts = [];
        } finally {
            filterLoading = false;
        }
    }

    function handlePinnedNodesChange(nodes: PinnedNode[]) {
        pinnedNodes = nodes;
        if (filterByPinned) {
            filterPromptsByPinned();
        }
    }

    function handleFilterToggle() {
        filterByPinned = !filterByPinned;
        if (filterByPinned && pinnedNodes.length > 0) {
            filterPromptsByPinned();
        }
    }

    async function tokenizeCustomPrompt() {
        if (!customPromptText.trim()) return;

        tokenizeLoading = true;
        customPromptError = null;
        tokenizedPreview = null;
        // Clear current prompt selection when entering custom text
        currentPromptId = null;
        cachedGraphs = [];
        activeGraphId = null;

        const t0 = performance.now();
        console.log(`[LocalAttr] tokenizeCustomPrompt: tokenizing ${customPromptText.length} chars...`);

        try {
            tokenizedPreview = await attrApi.tokenizeText(customPromptText);
            console.log(`[LocalAttr] tokenizeCustomPrompt: got ${tokenizedPreview.tokens.length} tokens in ${(performance.now() - t0).toFixed(0)}ms`);
        } catch (e) {
            customPromptError = e instanceof Error ? e.message : "Failed to tokenize";
            console.error(`[LocalAttr] tokenizeCustomPrompt FAILED (${(performance.now() - t0).toFixed(0)}ms):`, e);
        } finally {
            tokenizeLoading = false;
        }
    }

    function cancelCustomPrompt() {
        tokenizedPreview = null;
        customPromptText = "";
    }

    async function handleGeneratePrompts(nPrompts: number) {
        if (generatingPrompts) return;

        generatingPrompts = true;
        generateProgress = 0;
        generateCount = 0;
        generateError = null;

        const t0 = performance.now();
        console.log(`[LocalAttr] handleGeneratePrompts: starting generation of ${nPrompts} prompts...`);

        try {
            await attrApi.generatePrompts(
                { nPrompts },
                (progress, count) => {
                    generateProgress = progress;
                    generateCount = count;
                },
            );
            console.log(`[LocalAttr] handleGeneratePrompts: completed in ${(performance.now() - t0).toFixed(0)}ms`);
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
        <div class="no-run-message">
            <p>No run loaded. Select a run from the sidebar to view local attributions.</p>
            {#if serverError}
                <p class="server-error">{serverError}</p>
            {/if}
        </div>
    {:else}
        <!-- Sidebar: Prompt selection -->
        <div class="sidebar">
            <!-- Custom prompt input -->
            <div class="sidebar-section">
                <div class="custom-prompt-row">
                    <input
                        type="text"
                        bind:value={customPromptText}
                        placeholder="Enter custom text..."
                        onkeydown={(e) => e.key === "Enter" && tokenizeCustomPrompt()}
                        class="custom-input"
                        disabled={tokenizeLoading}
                    />
                    <button onclick={tokenizeCustomPrompt} disabled={!customPromptText.trim() || tokenizeLoading} class="btn-icon">
                        {tokenizeLoading ? "..." : "→"}
                    </button>
                </div>
                {#if customPromptError}
                    <div class="sidebar-error">{customPromptError}</div>
                {/if}
            </div>

            <!-- Filter by pinned -->
            {#if pinnedNodes.length > 0}
                <div class="sidebar-section filter-section">
                    <label class="filter-checkbox">
                        <input type="checkbox" checked={filterByPinned} onchange={handleFilterToggle} />
                        Filter by {pinnedNodes.length} pinned
                    </label>
                    {#if filterLoading}
                        <span class="filter-loading">...</span>
                    {/if}
                </div>
            {/if}

            <!-- Prompt list -->
            <div class="sidebar-section prompt-list-section">
                <div class="prompt-list-header">
                    Prompts ({displayedPrompts.length})
                </div>
                <div class="prompt-list">
                    {#each displayedPrompts as p}
                        <button
                            class="prompt-item"
                            class:selected={p.id === currentPromptId && !tokenizedPreview}
                            onclick={() => {
                                currentPromptId = p.id;
                                tokenizedPreview = null;
                                customPromptText = "";
                            }}
                        >
                            <span class="prompt-id">#{p.id}</span>
                            <span class="prompt-preview">{p.preview}</span>
                        </button>
                    {/each}
                    {#if displayedPrompts.length === 0}
                        <div class="prompt-list-empty">
                            {filterByPinned ? "No matching prompts" : "No prompts yet"}
                        </div>
                    {/if}
                </div>
                <div class="generate-section">
                    {#if generatingPrompts}
                        <div class="generate-progress">
                            <div class="mini-progress-bar">
                                <div class="mini-progress-fill" style="width: {generateProgress * 100}%"></div>
                            </div>
                            <span class="progress-label">{generateCount}</span>
                        </div>
                    {:else}
                        <button class="btn-generate" onclick={() => handleGeneratePrompts(100)} disabled={generatingPrompts}>
                            + Generate 100
                        </button>
                    {/if}
                    {#if generateError}
                        <div class="sidebar-error">{generateError}</div>
                    {/if}
                </div>
            </div>
        </div>

        <!-- Main content -->
        <div class="main-content">
            {#if activationContextsMissing}
                <div class="warning-banner">
                    Activation contexts not generated. Component hover info unavailable.
                </div>
            {/if}

            {#if promptError}
                <div class="error-banner">
                    {promptError}
                    <button onclick={() => computeGraph()}>Retry</button>
                </div>
            {/if}

            <!-- View controls bar (display options only) -->
            <div class="controls-bar">
                <span class="controls-label">View</span>
                <label>
                    <span>Top K</span>
                    <input type="number" bind:value={topK} min={10} max={10000} step={100} />
                </label>
                <label>
                    <span>Layout</span>
                    <select bind:value={nodeLayout}>
                        <option value="importance">Importance</option>
                        <option value="shuffled">Shuffled</option>
                        <option value="jittered">Jittered</option>
                    </select>
                </label>
            </div>

            <!-- Graph area with staging -->
            <div class="graph-container">
                <!-- Staged prompt header -->
                {#if stagedTokens || cachedGraphs.length > 0}
                    <div class="staged-header">
                        <!-- Staged tokens display -->
                        <div class="staged-tokens">
                            {#if stagedTokens}
                                {#each stagedTokens.tokens as tok}
                                    <span class="staged-token" class:custom={stagedTokens.isCustom}>{tok}</span>
                                {/each}
                            {:else if activeGraph}
                                {#each activeGraph.data.tokens as tok}
                                    <span class="staged-token">{tok}</span>
                                {/each}
                            {/if}
                            {#if stagedTokens?.isCustom}
                                <button class="btn-cancel" onclick={cancelCustomPrompt} title="Cancel">×</button>
                            {/if}
                        </div>

                        <!-- Computation options & actions -->
                        <div class="staged-controls">
                            <div class="compute-options">
                                <label>
                                    <span>Max CI</span>
                                    <input type="number" bind:value={maxMeanCI} min={0} max={1} step={0.01} />
                                </label>
                                <label class="checkbox">
                                    <input type="checkbox" bind:checked={normalizeEdges} />
                                    <span>Normalize</span>
                                </label>
                                <label class="checkbox">
                                    <input type="checkbox" bind:checked={useOptimized} />
                                    <span>Optimize</span>
                                </label>
                                {#if useOptimized}
                                    <label>
                                        <span>imp_min</span>
                                        <input type="number" bind:value={impMinCoeff} min={0.001} max={10} step={0.01} />
                                    </label>
                                    <label>
                                        <span>ce</span>
                                        <input type="number" bind:value={ceLossCoeff} min={0.001} max={10} step={0.1} />
                                    </label>
                                    <label>
                                        <span>steps</span>
                                        <input type="number" bind:value={optimSteps} min={10} max={5000} step={100} />
                                    </label>
                                    <label>
                                        <span>pnorm</span>
                                        <input type="number" bind:value={optimPnorm} min={0.1} max={1} step={0.1} />
                                    </label>
                                {/if}
                            </div>

                            <button
                                class="btn-compute"
                                onclick={computeGraph}
                                disabled={loadingPrompt || !stagedTokens}
                            >
                                {#if loadingPrompt}
                                    {useOptimized ? "Optimizing..." : "Computing..."}
                                {:else}
                                    Compute{useOptimized ? " (Optimized)" : ""}
                                {/if}
                            </button>
                        </div>

                        <!-- Graph tabs -->
                        {#if cachedGraphs.length > 0}
                            <div class="graph-tabs">
                                {#each cachedGraphs as graph}
                                    <div class="graph-tab" class:active={graph.id === activeGraphId}>
                                        <button class="tab-label" onclick={() => (activeGraphId = graph.id)}>
                                            {graph.label}
                                        </button>
                                        <button class="tab-close" onclick={() => closeGraphTab(graph.id)}>×</button>
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    </div>
                {/if}

                <!-- Optimization results banner -->
                {#if activeGraph?.data.optimization}
                    <div class="optim-results">
                        <span><strong>Target:</strong> "{activeGraph.data.optimization.label_str}" @ {(activeGraph.data.optimization.label_prob * 100).toFixed(1)}%</span>
                        <span><strong>L0:</strong> {activeGraph.data.optimization.l0_total.toFixed(0)} active</span>
                    </div>
                {/if}

                <!-- Graph display area -->
                <div class="graph-area" class:loading={loadingPrompt}>
                    {#if loadingPrompt}
                        <div class="loading-overlay">
                            {#if loadingProgress}
                                <div class="progress-container">
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {(loadingProgress.current / loadingProgress.total) * 100}%"></div>
                                    </div>
                                    <span class="progress-text">Computing {loadingProgress.stage}... ({loadingProgress.current}/{loadingProgress.total})</span>
                                </div>
                            {:else}
                                <div class="loading-spinner"></div>
                                <span>{loadingMode === "optimized" ? "Running optimization..." : "Computing graph..."}</span>
                            {/if}
                        </div>
                    {/if}

                    {#if activeGraph}
                        <LocalAttributionsGraph
                            data={activeGraph.data}
                            {topK}
                            {nodeLayout}
                            {activationContextsSummary}
                            {pinnedNodes}
                            onPinnedNodesChange={handlePinnedNodesChange}
                        />
                    {:else if stagedTokens && !loadingPrompt}
                        <div class="empty-state">
                            <p>Click <strong>Compute</strong> to generate the attribution graph</p>
                        </div>
                    {:else if !loadingPrompt}
                        <div class="empty-state">
                            <p>Select a prompt or enter custom text</p>
                            <p class="hint">{prompts.length} prompts available</p>
                        </div>
                    {/if}
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .local-attributions-tab {
        display: flex;
        flex: 1;
        min-height: 0;
        background: #f5f5f5;
    }

    /* Sidebar */
    .sidebar {
        width: 280px;
        background: #fff;
        border-right: 1px solid #e0e0e0;
        display: flex;
        flex-direction: column;
        flex-shrink: 0;
    }

    .sidebar-section {
        padding: 0.75rem;
    }

    .sidebar-error {
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: #ffebee;
        color: #c62828;
        font-size: 0.8rem;
        border-radius: 4px;
    }

    .custom-prompt-row {
        display: flex;
        gap: 0.5rem;
    }

    .custom-input {
        flex: 1;
        padding: 0.5rem 0.75rem;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        font-size: 0.875rem;
        background: #fafafa;
        transition: border-color 0.15s, background 0.15s;
    }

    .custom-input:focus {
        outline: none;
        border-color: #2196f3;
        background: #fff;
    }

    .btn-icon {
        padding: 0.5rem 0.75rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 600;
        transition: background 0.15s;
    }

    .btn-icon:hover:not(:disabled) {
        background: #1976d2;
    }

    .btn-icon:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
    }

    .filter-section {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #fff3e0;
        border-top: 1px solid #e0e0e0;
    }

    .filter-checkbox {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.85rem;
        cursor: pointer;
    }

    .filter-loading {
        font-size: 0.8rem;
        color: #999;
    }

    .prompt-list-section {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        border-top: 1px solid #e0e0e0;
        padding: 0;
    }

    .prompt-list-header {
        padding: 0.5rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #666;
        background: #fafafa;
        border-bottom: 1px solid #e0e0e0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .prompt-list {
        flex: 1;
        overflow-y: auto;
        min-height: 0;
    }

    .prompt-item {
        width: 100%;
        padding: 0.5rem 0.75rem;
        background: transparent;
        border: none;
        border-bottom: 1px solid #f0f0f0;
        cursor: pointer;
        text-align: left;
        display: flex;
        gap: 0.5rem;
        align-items: baseline;
        transition: background 0.1s;
    }

    .prompt-item:hover {
        background: #f5f5f5;
    }

    .prompt-item.selected {
        background: #e3f2fd;
    }

    .prompt-id {
        font-size: 0.7rem;
        color: #9e9e9e;
        flex-shrink: 0;
    }

    .prompt-preview {
        font-family: "SF Mono", Monaco, monospace;
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #424242;
    }

    .prompt-list-empty {
        padding: 1.5rem;
        text-align: center;
        color: #9e9e9e;
        font-size: 0.85rem;
    }

    .generate-section {
        padding: 0.5rem 0.75rem;
        border-top: 1px solid #e0e0e0;
        background: #fafafa;
    }

    .btn-generate {
        width: 100%;
        padding: 0.5rem;
        background: #4caf50;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        transition: background 0.15s;
    }

    .btn-generate:hover:not(:disabled) {
        background: #43a047;
    }

    .btn-generate:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
    }

    .generate-progress {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .mini-progress-bar {
        flex: 1;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
    }

    .mini-progress-fill {
        height: 100%;
        background: #4caf50;
        transition: width 0.1s ease;
    }

    .progress-label {
        font-size: 0.75rem;
        color: #666;
        min-width: 30px;
    }

    /* Main content */
    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 0;
        padding: 1rem;
        gap: 0.75rem;
    }

    .controls-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 1rem;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    .controls-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #9e9e9e;
    }

    .controls-bar label {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: #616161;
    }

    .controls-bar label span {
        font-weight: 500;
    }

    .controls-bar input[type="number"] {
        width: 60px;
        padding: 0.25rem 0.4rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    .controls-bar select {
        padding: 0.25rem 0.4rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    /* Graph container */
    .graph-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        overflow: hidden;
    }

    .staged-header {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background: #fafafa;
        border-bottom: 1px solid #e0e0e0;
    }

    .staged-tokens {
        display: flex;
        flex-wrap: wrap;
        gap: 2px;
        align-items: center;
    }

    .staged-token {
        padding: 2px 6px;
        background: #e8e8e8;
        border-radius: 3px;
        font-family: "SF Mono", Monaco, monospace;
        font-size: 0.8rem;
        color: #424242;
    }

    .staged-token.custom {
        background: #e3f2fd;
        color: #1565c0;
    }

    .btn-cancel {
        margin-left: 0.5rem;
        padding: 0 0.4rem;
        background: transparent;
        border: 1px solid #bdbdbd;
        border-radius: 3px;
        color: #757575;
        cursor: pointer;
        font-size: 0.9rem;
        line-height: 1.2;
    }

    .btn-cancel:hover {
        background: #f5f5f5;
        border-color: #9e9e9e;
    }

    .staged-controls {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .compute-options {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .compute-options label {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.8rem;
        color: #616161;
    }

    .compute-options label span {
        font-weight: 500;
    }

    .compute-options input[type="number"] {
        width: 55px;
        padding: 0.2rem 0.35rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    .compute-options label.checkbox {
        gap: 0.2rem;
    }

    .graph-tabs {
        display: flex;
        gap: 0.25rem;
    }

    .graph-tab {
        display: flex;
        align-items: center;
        background: #e0e0e0;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #616161;
        transition: background 0.1s;
    }

    .graph-tab:hover {
        background: #d5d5d5;
    }

    .graph-tab.active {
        background: #2196f3;
        color: white;
    }

    .tab-label {
        padding: 0.35rem 0.5rem;
        background: transparent;
        border: none;
        font-size: inherit;
        color: inherit;
        cursor: pointer;
    }

    .tab-close {
        padding: 0.35rem 0.4rem;
        background: transparent;
        border: none;
        font-size: 0.85rem;
        line-height: 1;
        opacity: 0.6;
        cursor: pointer;
        color: inherit;
        border-left: 1px solid rgba(0, 0, 0, 0.1);
    }

    .graph-tab.active .tab-close {
        border-left-color: rgba(255, 255, 255, 0.3);
    }

    .tab-close:hover {
        opacity: 1;
        background: rgba(0, 0, 0, 0.1);
    }

    .btn-compute {
        padding: 0.5rem 1rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        transition: background 0.15s;
        white-space: nowrap;
    }

    .btn-compute:hover:not(:disabled) {
        background: #1976d2;
    }

    .btn-compute:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
    }

    .optim-results {
        display: flex;
        gap: 1.5rem;
        padding: 0.5rem 1rem;
        background: #fff3e0;
        font-size: 0.8rem;
        color: #e65100;
    }

    .graph-area {
        flex: 1;
        position: relative;
        min-height: 400px;
        overflow: hidden;
    }

    .graph-area.loading {
        opacity: 0.6;
    }

    .loading-overlay {
        position: absolute;
        inset: 0;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        z-index: 100;
    }

    .loading-spinner {
        width: 28px;
        height: 28px;
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

    .progress-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        width: 280px;
    }

    .progress-bar {
        width: 100%;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: #2196f3;
        transition: width 0.15s ease-out;
    }

    .progress-text {
        font-size: 0.8rem;
        color: #757575;
    }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #757575;
        text-align: center;
    }

    .empty-state p {
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }

    .empty-state .hint {
        font-size: 0.8rem;
        color: #9e9e9e;
    }

    /* Banners */
    .no-run-message {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: #757575;
        text-align: center;
        padding: 2rem;
    }

    .no-run-message .server-error {
        color: #c62828;
        margin-top: 0.5rem;
    }

    .warning-banner {
        padding: 0.6rem 1rem;
        background: #fff3e0;
        border: 1px solid #ffcc80;
        border-radius: 6px;
        color: #e65100;
        font-size: 0.85rem;
    }

    .error-banner {
        padding: 0.6rem 1rem;
        background: #ffebee;
        border: 1px solid #ef9a9a;
        border-radius: 6px;
        color: #c62828;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.85rem;
    }

    .error-banner button {
        margin-left: auto;
        padding: 0.25rem 0.6rem;
        background: #ef5350;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.8rem;
    }

    .error-banner button:hover {
        background: #e53935;
    }
</style>
