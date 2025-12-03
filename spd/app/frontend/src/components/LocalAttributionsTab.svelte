<script lang="ts">
    import * as attrApi from "../lib/localAttributionsApi";
    import * as mainApi from "../lib/api";
    import type {
        PromptPreview,
        PromptData,
        ActivationContextsSummary,
        PinnedNode,
    } from "../lib/localAttributionsTypes";
    import LocalAttributionsGraph from "./LocalAttributionsGraph.svelte";

    // Types
    type CachedGraph = { id: string; label: string; data: PromptData };
    type PromptCard = {
        id: string;
        promptId: number | null;
        tokens: string[];
        tokenIds: number[] | null;
        isCustom: boolean;
        graphs: CachedGraph[];
        activeGraphId: string | null;
    };

    // Server state
    let loadedRun = $state<mainApi.LoadedRun | null>(null);
    let serverError = $state<string | null>(null);

    // Available prompts (for picker)
    let prompts = $state<PromptPreview[]>([]);

    // Prompt cards state
    let promptCards = $state<PromptCard[]>([]);
    let activeCardId = $state<string | null>(null);

    // Prompt picker state
    let showPromptPicker = $state(false);
    let customPromptText = $state("");
    let tokenizeLoading = $state(false);
    let filterByPinned = $state(false);
    let filteredPrompts = $state<PromptPreview[]>([]);
    let filterLoading = $state(false);

    // Loading state (per card, but we track globally for simplicity)
    let loadingCardId = $state<string | null>(null);
    let loadingMode = $state<"standard" | "optimized">("standard");
    let loadingProgress = $state<{ current: number; total: number; stage: string } | null>(null);
    let computeError = $state<string | null>(null);

    // Prompt generation state
    let generatingPrompts = $state(false);
    let generateProgress = $state(0);
    let generateCount = $state(0);

    // Activation contexts
    let activationContextsSummary = $state<ActivationContextsSummary | null>(null);
    let activationContextsMissing = $state(false);

    // View controls (global)
    let topK = $state(800);
    let nodeLayout = $state<"importance" | "shuffled" | "jittered">("importance");

    // Compute controls (per-computation, could be per-card but keeping simple)
    let maxMeanCI = $state(1.0);
    let normalizeEdges = $state(true);
    let useOptimized = $state(false);
    let impMinCoeff = $state(0.1);
    let ceLossCoeff = $state(1.0);
    let optimSteps = $state(500);
    let optimPnorm = $state(0.3);

    // Pinned nodes (for search)
    let pinnedNodes = $state<PinnedNode[]>([]);

    // Derived
    const activeCard = $derived(promptCards.find((c) => c.id === activeCardId) ?? null);
    const activeGraph = $derived(activeCard?.graphs.find((g) => g.id === activeCard.activeGraphId) ?? null);
    const displayedPrompts = $derived(filterByPinned ? filteredPrompts : prompts);

    // Load server status on mount
    $effect(() => {
        loadServerStatus();
    });

    // When loaded run changes, refresh data
    let previousRunId: number | null = null;
    $effect(() => {
        const currentRunId = loadedRun?.id ?? null;
        if (currentRunId !== null && currentRunId !== previousRunId) {
            previousRunId = currentRunId;
            loadPromptsList();
            loadActivationContextsSummary();
            promptCards = [];
            activeCardId = null;
        } else if (currentRunId === null && previousRunId !== null) {
            previousRunId = null;
            prompts = [];
            promptCards = [];
            activeCardId = null;
            activationContextsSummary = null;
            activationContextsMissing = false;
        }
    });

    async function loadServerStatus() {
        try {
            loadedRun = await mainApi.getStatus();
            serverError = null;
        } catch (e) {
            serverError = e instanceof Error ? e.message : "Failed to connect to server";
        }
    }

    async function loadPromptsList() {
        try {
            prompts = await attrApi.listPrompts();
        } catch (e) {
            console.error("[LocalAttr] loadPromptsList FAILED:", e);
        }
    }

    async function loadActivationContextsSummary() {
        try {
            activationContextsSummary = await attrApi.getActivationContextsSummary();
            activationContextsMissing = false;
        } catch (e) {
            const status = (e as { status?: number }).status;
            if (status === 404) {
                activationContextsMissing = true;
                activationContextsSummary = null;
            }
        }
    }

    function addPromptCard(promptId: number | null, tokens: string[], tokenIds: number[] | null, isCustom: boolean) {
        const cardId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        const newCard: PromptCard = {
            id: cardId,
            promptId,
            tokens,
            tokenIds,
            isCustom,
            graphs: [],
            activeGraphId: null,
        };
        promptCards = [...promptCards, newCard];
        activeCardId = cardId;
        showPromptPicker = false;
        customPromptText = "";
    }

    function selectPromptFromPicker(prompt: PromptPreview) {
        addPromptCard(prompt.id, prompt.tokens, null, false);
    }

    async function addCustomPrompt() {
        if (!customPromptText.trim()) return;

        tokenizeLoading = true;
        try {
            const result = await attrApi.tokenizeText(customPromptText);
            addPromptCard(null, result.tokens, result.token_ids, true);
        } catch (e) {
            console.error("[LocalAttr] tokenize FAILED:", e);
        } finally {
            tokenizeLoading = false;
        }
    }

    function closeCard(cardId: string) {
        promptCards = promptCards.filter((c) => c.id !== cardId);
        if (activeCardId === cardId) {
            activeCardId = promptCards.length > 0 ? promptCards[promptCards.length - 1].id : null;
        }
    }

    function closeGraphTab(cardId: string, graphId: string) {
        promptCards = promptCards.map((card) => {
            if (card.id !== cardId) return card;
            const newGraphs = card.graphs.filter((g) => g.id !== graphId);
            return {
                ...card,
                graphs: newGraphs,
                activeGraphId: card.activeGraphId === graphId
                    ? (newGraphs.length > 0 ? newGraphs[newGraphs.length - 1].id : null)
                    : card.activeGraphId,
            };
        });
    }

    function setActiveGraph(cardId: string, graphId: string) {
        promptCards = promptCards.map((card) =>
            card.id === cardId ? { ...card, activeGraphId: graphId } : card
        );
    }

    async function computeGraph() {
        if (!activeCard || loadingCardId) return;

        loadingCardId = activeCard.id;
        loadingMode = useOptimized ? "optimized" : "standard";
        loadingProgress = null;
        computeError = null;

        try {
            let data: PromptData;

            if (activeCard.isCustom && activeCard.tokenIds) {
                data = await attrApi.computeCustomPrompt({
                    tokenIds: activeCard.tokenIds,
                    normalize: normalizeEdges,
                });
            } else if (activeCard.promptId !== null) {
                if (useOptimized) {
                    data = await attrApi.getPromptOptimizedStreaming(
                        activeCard.promptId,
                        {
                            maxMeanCI,
                            normalize: normalizeEdges,
                            impMinCoeff,
                            ceLossCoeff,
                            steps: optimSteps,
                            pnorm: optimPnorm,
                        },
                        (progress) => {
                            loadingProgress = progress;
                        },
                    );
                } else {
                    data = await attrApi.getPromptStreaming(
                        activeCard.promptId,
                        { maxMeanCI, normalize: normalizeEdges },
                        (progress) => {
                            loadingProgress = progress;
                        },
                    );
                }
            } else {
                throw new Error("Invalid card state");
            }

            const graphId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
            const label = useOptimized ? `Optimized (${optimSteps} steps)` : "Standard";

            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: [...card.graphs, { id: graphId, label, data }],
                    activeGraphId: graphId,
                };
            });
        } catch (e) {
            computeError = e instanceof Error ? e.message : "Failed to compute graph";
        } finally {
            loadingCardId = null;
            loadingProgress = null;
        }
    }

    async function filterPromptsByPinned() {
        if (pinnedNodes.length === 0) {
            filteredPrompts = [];
            return;
        }

        filterLoading = true;
        try {
            const components = pinnedNodes.map((p) => `${p.layer}:${p.cIdx}`);
            const result = await attrApi.searchPrompts(components, "all");
            filteredPrompts = result.results;
        } catch (e) {
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

    async function handleGeneratePrompts(nPrompts: number) {
        if (generatingPrompts) return;

        generatingPrompts = true;
        generateProgress = 0;
        generateCount = 0;

        try {
            await attrApi.generatePrompts(
                { nPrompts },
                (progress, count) => {
                    generateProgress = progress;
                    generateCount = count;
                },
            );
            await loadPromptsList();
        } catch (e) {
            console.error("[LocalAttr] generatePrompts FAILED:", e);
        } finally {
            generatingPrompts = false;
        }
    }

    function getCardLabel(card: PromptCard): string {
        if (card.isCustom) {
            return card.tokens.slice(0, 3).join("") + (card.tokens.length > 3 ? "..." : "");
        }
        return `#${card.promptId}`;
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
        <div class="main-content">
            {#if activationContextsMissing}
                <div class="warning-banner">
                    Activation contexts not generated. Component hover info unavailable.
                </div>
            {/if}

            <!-- View controls -->
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

            <!-- Prompt cards container -->
            <div class="graph-container">
                <!-- Prompt card tabs -->
                <div class="card-tabs-row">
                    <div class="card-tabs">
                        {#each promptCards as card}
                            <div class="card-tab" class:active={card.id === activeCardId}>
                                <button class="card-tab-label" onclick={() => (activeCardId = card.id)}>
                                    {getCardLabel(card)}
                                </button>
                                <button class="card-tab-close" onclick={() => closeCard(card.id)}>×</button>
                            </div>
                        {/each}
                    </div>

                    <!-- Add prompt button -->
                    <div class="add-prompt-wrapper">
                        <button class="btn-add-prompt" onclick={() => (showPromptPicker = !showPromptPicker)}>
                            + Add Prompt
                        </button>

                        {#if showPromptPicker}
                            <div class="prompt-picker">
                                <div class="picker-header">
                                    <input
                                        type="text"
                                        bind:value={customPromptText}
                                        placeholder="Enter custom text..."
                                        onkeydown={(e) => e.key === "Enter" && addCustomPrompt()}
                                        class="picker-input"
                                    />
                                    <button
                                        onclick={addCustomPrompt}
                                        disabled={!customPromptText.trim() || tokenizeLoading}
                                        class="btn-tokenize"
                                    >
                                        {tokenizeLoading ? "..." : "Add"}
                                    </button>
                                </div>

                                <div class="picker-filter">
                                    <label class="filter-checkbox">
                                        <input
                                            type="checkbox"
                                            checked={filterByPinned}
                                            onchange={handleFilterToggle}
                                            disabled={pinnedNodes.length === 0}
                                        />
                                        Filter by pinned ({pinnedNodes.length})
                                    </label>
                                    {#if filterLoading}
                                        <span class="filter-loading">...</span>
                                    {/if}
                                </div>

                                <div class="picker-list">
                                    {#each displayedPrompts as p}
                                        <button class="picker-item" onclick={() => selectPromptFromPicker(p)}>
                                            <span class="picker-item-id">#{p.id}</span>
                                            <span class="picker-item-preview">{p.preview}</span>
                                        </button>
                                    {/each}
                                    {#if displayedPrompts.length === 0}
                                        <div class="picker-empty">
                                            {filterByPinned ? "No matching prompts" : "No prompts yet"}
                                        </div>
                                    {/if}
                                </div>

                                <div class="picker-footer">
                                    {#if generatingPrompts}
                                        <div class="generate-progress">
                                            <div class="mini-progress-bar">
                                                <div class="mini-progress-fill" style="width: {generateProgress * 100}%"></div>
                                            </div>
                                            <span>{generateCount}</span>
                                        </div>
                                    {:else}
                                        <button class="btn-generate" onclick={() => handleGeneratePrompts(100)}>
                                            + Generate 100
                                        </button>
                                    {/if}
                                </div>
                            </div>
                        {/if}
                    </div>
                </div>

                {#if activeCard}
                    <!-- Staged header for active card -->
                    <div class="staged-header">
                        <div class="staged-tokens">
                            {#each activeCard.tokens as tok}
                                <span class="staged-token" class:custom={activeCard.isCustom}>{tok}</span>
                            {/each}
                        </div>

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
                                disabled={loadingCardId !== null}
                            >
                                {#if loadingCardId === activeCard.id}
                                    {loadingMode === "optimized" ? "Optimizing..." : "Computing..."}
                                {:else}
                                    Compute{useOptimized ? " (Optimized)" : ""}
                                {/if}
                            </button>
                        </div>

                        {#if activeCard.graphs.length > 0}
                            <div class="graph-tabs">
                                {#each activeCard.graphs as graph}
                                    <div class="graph-tab" class:active={graph.id === activeCard.activeGraphId}>
                                        <button class="tab-label" onclick={() => setActiveGraph(activeCard.id, graph.id)}>
                                            {graph.label}
                                        </button>
                                        <button class="tab-close" onclick={() => closeGraphTab(activeCard.id, graph.id)}>×</button>
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    </div>

                    {#if activeGraph?.data.optimization}
                        <div class="optim-results">
                            <span><strong>Target:</strong> "{activeGraph.data.optimization.label_str}" @ {(activeGraph.data.optimization.label_prob * 100).toFixed(1)}%</span>
                            <span><strong>L0:</strong> {activeGraph.data.optimization.l0_total.toFixed(0)} active</span>
                        </div>
                    {/if}

                    {#if computeError}
                        <div class="error-banner">
                            {computeError}
                            <button onclick={() => computeGraph()}>Retry</button>
                        </div>
                    {/if}

                    <div class="graph-area" class:loading={loadingCardId === activeCard.id}>
                        {#if loadingCardId === activeCard.id}
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
                            {#key activeGraph.id}
                                <LocalAttributionsGraph
                                    data={activeGraph.data}
                                    {topK}
                                    {nodeLayout}
                                    {activationContextsSummary}
                                    {pinnedNodes}
                                    onPinnedNodesChange={handlePinnedNodesChange}
                                />
                            {/key}
                        {:else if !loadingCardId}
                            <div class="empty-state">
                                <p>Click <strong>Compute</strong> to generate the attribution graph</p>
                            </div>
                        {/if}
                    </div>
                {:else}
                    <div class="empty-state full">
                        <p>Click <strong>+ Add Prompt</strong> to get started</p>
                        <p class="hint">{prompts.length} prompts available</p>
                    </div>
                {/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .local-attributions-tab {
        display: flex;
        flex: 1;
        min-height: 0;
        /* background: #f5f5f5; */
    }

    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 0;
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

    /* Card tabs row */
    .card-tabs-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: #f5f5f5;
        border-bottom: 1px solid #e0e0e0;
    }

    .card-tabs {
        display: flex;
        gap: 0.25rem;
        flex: 1;
        overflow-x: auto;
    }

    .card-tab {
        display: flex;
        align-items: center;
        background: #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #616161;
        flex-shrink: 0;
    }

    .card-tab:hover {
        background: #d5d5d5;
    }

    .card-tab.active {
        background: #fff;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .card-tab-label {
        padding: 0.4rem 0.6rem;
        background: transparent;
        border: none;
        font-size: inherit;
        color: inherit;
        cursor: pointer;
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .card-tab-close {
        padding: 0.4rem 0.4rem;
        background: transparent;
        border: none;
        font-size: 0.85rem;
        line-height: 1;
        opacity: 0.5;
        cursor: pointer;
        color: inherit;
    }

    .card-tab-close:hover {
        opacity: 1;
    }

    /* Add prompt button & picker */
    .add-prompt-wrapper {
        position: relative;
    }

    .btn-add-prompt {
        padding: 0.4rem 0.75rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        white-space: nowrap;
    }

    .btn-add-prompt:hover {
        background: #1976d2;
    }

    .prompt-picker {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 0.5rem;
        width: 320px;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        z-index: 1000;
        overflow: hidden;
    }

    .picker-header {
        display: flex;
        gap: 0.5rem;
        padding: 0.75rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .picker-input {
        flex: 1;
        padding: 0.5rem 0.75rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .picker-input:focus {
        outline: none;
        border-color: #2196f3;
    }

    .btn-tokenize {
        padding: 0.5rem 0.75rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 0.85rem;
        cursor: pointer;
    }

    .btn-tokenize:hover:not(:disabled) {
        background: #1976d2;
    }

    .btn-tokenize:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
    }

    .picker-filter {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: #fafafa;
        border-bottom: 1px solid #e0e0e0;
    }

    .filter-checkbox {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.8rem;
        color: #616161;
        cursor: pointer;
    }

    .filter-loading {
        font-size: 0.75rem;
        color: #999;
    }

    .picker-list {
        max-height: 240px;
        overflow-y: auto;
    }

    .picker-item {
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
    }

    .picker-item:hover {
        background: #f5f5f5;
    }

    .picker-item-id {
        font-size: 0.7rem;
        color: #9e9e9e;
        flex-shrink: 0;
    }

    .picker-item-preview {
        font-family: "SF Mono", Monaco, monospace;
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #424242;
    }

    .picker-empty {
        padding: 1.5rem;
        text-align: center;
        color: #9e9e9e;
        font-size: 0.85rem;
    }

    .picker-footer {
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
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
    }

    .btn-generate:hover {
        background: #43a047;
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

    /* Staged header */
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

    .btn-compute {
        padding: 0.5rem 1rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        white-space: nowrap;
    }

    .btn-compute:hover:not(:disabled) {
        background: #1976d2;
    }

    .btn-compute:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
    }

    /* Graph tabs */
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
    }

    /* Optimization results */
    .optim-results {
        display: flex;
        gap: 1.5rem;
        padding: 0.5rem 1rem;
        background: #fff3e0;
        font-size: 0.8rem;
        color: #e65100;
    }

    /* Graph area */
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
        padding: 2rem;
    }

    .empty-state.full {
        flex: 1;
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
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.85rem;
        color: #c62828;
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
