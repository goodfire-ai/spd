<script lang="ts">
    import * as attrApi from "../lib/localAttributionsApi";
    import * as mainApi from "../lib/api";
    import type {
        PromptPreview,
        GraphData,
        ActivationContextsSummary,
        PinnedNode,
    } from "../lib/localAttributionsTypes";
    import type { PromptCard, ComputeOptions, OptimizeConfig, LoadingState } from "./local-attr/types";
    import LocalAttributionsGraph from "./LocalAttributionsGraph.svelte";
    import ViewControls from "./local-attr/ViewControls.svelte";
    import PromptPicker from "./local-attr/PromptPicker.svelte";
    import PromptCardTabs from "./local-attr/PromptCardTabs.svelte";
    import PromptCardHeader from "./local-attr/PromptCardHeader.svelte";
    import ComputeProgressOverlay from "./local-attr/ComputeProgressOverlay.svelte";

    // Server state
    let loadedRun = $state<mainApi.LoadedRun | null>(null);
    let serverError = $state<string | null>(null);

    // Available prompts (for picker)
    let prompts = $state<PromptPreview[]>([]);

    // Prompt cards state
    let promptCards = $state<PromptCard[]>([]);
    let activeCardId = $state<string | null>(null);

    // Prompt picker state
    let filterByPinned = $state(false);
    let filteredPrompts = $state<PromptPreview[]>([]);
    let filterLoading = $state(false);

    // Loading state
    let loadingCardId = $state<string | null>(null);
    let loadingState = $state<LoadingState | null>(null);
    let computeError = $state<string | null>(null);

    // Graph generation state
    let generatingGraphs = $state(false);
    let generateProgress = $state(0);
    let generateCount = $state(0);

    // Activation contexts
    let activationContextsSummary = $state<ActivationContextsSummary | null>(null);
    let activationContextsMissing = $state(false);

    // View controls
    let topK = $state(800);
    let nodeLayout = $state<"importance" | "shuffled" | "jittered">("importance");
    let filteredEdgeCount = $state<number | null>(null);

    // Compute options
    let computeOptions = $state<ComputeOptions>({
        maxMeanCI: 1.0,
        normalizeEdges: true,
        useOptimized: false,
        optimizeConfig: {
            labelTokenText: "",
            labelTokenId: null,
            labelTokenPreview: null,
            impMinCoeff: 0.1,
            ceLossCoeff: 1.0,
            steps: 2000,
            pnorm: 0.3,
        },
    });

    // Pinned nodes (for search)
    let pinnedNodes = $state<PinnedNode[]>([]);

    // Tokenize label text when it changes
    let labelTokenizeTimeout: ReturnType<typeof setTimeout> | null = null;
    $effect(() => {
        const text = computeOptions.optimizeConfig.labelTokenText.trim();
        if (!text) {
            computeOptions.optimizeConfig.labelTokenId = null;
            computeOptions.optimizeConfig.labelTokenPreview = null;
            return;
        }

        if (labelTokenizeTimeout) clearTimeout(labelTokenizeTimeout);
        labelTokenizeTimeout = setTimeout(async () => {
            try {
                const result = await attrApi.tokenizeText(text);
                if (result.token_ids.length === 1) {
                    computeOptions.optimizeConfig.labelTokenId = result.token_ids[0];
                    computeOptions.optimizeConfig.labelTokenPreview = result.tokens[0];
                } else if (result.token_ids.length > 1) {
                    computeOptions.optimizeConfig.labelTokenId = result.token_ids[0];
                    computeOptions.optimizeConfig.labelTokenPreview = `${result.tokens[0]} (${result.token_ids.length} tokens, using first)`;
                } else {
                    computeOptions.optimizeConfig.labelTokenId = null;
                    computeOptions.optimizeConfig.labelTokenPreview = "(no tokens)";
                }
            } catch {
                computeOptions.optimizeConfig.labelTokenId = null;
                computeOptions.optimizeConfig.labelTokenPreview = "(error)";
            }
        }, 300);
    });

    // Derived state
    const activeCard = $derived(promptCards.find((c) => c.id === activeCardId) ?? null);
    const activeGraph = $derived.by(() => {
        const card = promptCards.find((c) => c.id === activeCardId);
        if (!card) return null;
        return card.graphs.find((g) => g.id === card.activeGraphId) ?? null;
    });

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
    }

    function handleSelectPrompt(prompt: PromptPreview) {
        addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, false);
    }

    async function handleAddCustomPrompt(text: string) {
        const result = await attrApi.tokenizeText(text);
        addPromptCard(null, result.tokens, result.token_ids, true);
    }

    function handleCloseCard(cardId: string) {
        promptCards = promptCards.filter((c) => c.id !== cardId);
        if (activeCardId === cardId) {
            activeCardId = promptCards.length > 0 ? promptCards[promptCards.length - 1].id : null;
        }
    }

    function handleSelectGraph(graphId: string) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) =>
            card.id === activeCard.id ? { ...card, activeGraphId: graphId } : card,
        );
    }

    function handleCloseGraph(graphId: string) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            const newGraphs = card.graphs.filter((g) => g.id !== graphId);
            return {
                ...card,
                graphs: newGraphs,
                activeGraphId:
                    card.activeGraphId === graphId
                        ? newGraphs.length > 0
                            ? newGraphs[newGraphs.length - 1].id
                            : null
                        : card.activeGraphId,
            };
        });
    }

    function handleOptionsChange(partial: Partial<ComputeOptions>) {
        computeOptions = { ...computeOptions, ...partial };
    }

    function handleOptimizeConfigChange(partial: Partial<OptimizeConfig>) {
        computeOptions.optimizeConfig = { ...computeOptions.optimizeConfig, ...partial };
    }

    async function computeGraphForCard() {
        if (!activeCard || !activeCard.tokenIds || loadingCardId) return;

        loadingCardId = activeCard.id;
        computeError = null;

        const optConfig = computeOptions.optimizeConfig;
        const isOptimized = computeOptions.useOptimized;

        // Set up stages - optimized has 2 stages, standard has 1
        if (isOptimized) {
            loadingState = {
                stages: [
                    { name: "Optimizing", progress: 0 },
                    { name: "Computing graph", progress: 0 },
                ],
                currentStage: 0,
            };
        } else {
            loadingState = {
                stages: [{ name: "Computing graph", progress: 0 }],
                currentStage: 0,
            };
        }

        try {
            let data: GraphData;

            if (isOptimized) {
                if (!optConfig.labelTokenId) throw new Error("Label token required for optimization");
                data = await attrApi.computeGraphOptimizedStreaming(
                    {
                        tokenIds: activeCard.tokenIds,
                        labelToken: optConfig.labelTokenId,
                        normalize: computeOptions.normalizeEdges,
                        impMinCoeff: optConfig.impMinCoeff,
                        ceLossCoeff: optConfig.ceLossCoeff,
                        steps: optConfig.steps,
                        pnorm: optConfig.pnorm,
                        outputProbThreshold: 0.01,
                    },
                    (progress) => {
                        if (!loadingState) return;
                        if (progress.stage === "graph") {
                            // Graph computation stage
                            loadingState.currentStage = 1;
                            loadingState.stages[1].progress = progress.current / progress.total;
                        } else {
                            // Optimization stage
                            loadingState.stages[0].progress = progress.current / progress.total;
                        }
                    },
                );
            } else {
                data = await attrApi.computeGraphStreaming(
                    {
                        tokenIds: activeCard.tokenIds,
                        normalize: computeOptions.normalizeEdges,
                    },
                    (progress) => {
                        if (!loadingState) return;
                        loadingState.stages[0].progress = progress.current / progress.total;
                    },
                );
            }

            const graphId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
            const label = isOptimized ? `Optimized (${optConfig.steps} steps)` : "Standard";

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
            loadingState = null;
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
        } catch {
            filteredPrompts = [];
        } finally {
            filterLoading = false;
        }
    }

    function handleFilterToggle() {
        filterByPinned = !filterByPinned;
        if (filterByPinned && pinnedNodes.length > 0) {
            filterPromptsByPinned();
        }
    }

    function handlePinnedNodesChange(nodes: PinnedNode[]) {
        pinnedNodes = nodes;
        if (filterByPinned) {
            filterPromptsByPinned();
        }
    }

    async function handleGeneratePrompts(nPrompts: number) {
        if (generatingGraphs) return;

        generatingGraphs = true;
        generateProgress = 0;
        generateCount = 0;

        try {
            await attrApi.generatePrompts({ nPrompts }, (progress: number, count: number) => {
                generateProgress = progress;
                generateCount = count;
            });
            await loadPromptsList();
        } catch (e) {
            console.error("[LocalAttr] generatePrompts FAILED:", e);
        } finally {
            generatingGraphs = false;
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
        <div class="main-content">
            {#if activationContextsMissing}
                <div class="warning-banner">Activation contexts not generated. Component hover info unavailable.</div>
            {/if}

            <ViewControls
                {topK}
                {nodeLayout}
                {filteredEdgeCount}
                onTopKChange={(v) => (topK = v)}
                onLayoutChange={(v) => (nodeLayout = v)}
            />

            <div class="graph-container">
                <div class="card-tabs-row">
                    <PromptCardTabs
                        cards={promptCards}
                        {activeCardId}
                        onSelectCard={(id) => (activeCardId = id)}
                        onCloseCard={handleCloseCard}
                    />
                    <PromptPicker
                        {prompts}
                        {filteredPrompts}
                        {pinnedNodes}
                        {filterByPinned}
                        {filterLoading}
                        {generatingGraphs}
                        {generateProgress}
                        {generateCount}
                        onSelectPrompt={handleSelectPrompt}
                        onAddCustom={handleAddCustomPrompt}
                        onFilterToggle={handleFilterToggle}
                        onGenerate={handleGeneratePrompts}
                    />
                </div>

                {#if activeCard}
                    <PromptCardHeader
                        card={activeCard}
                        options={computeOptions}
                        isLoading={loadingCardId === activeCard.id}
                        onOptionsChange={handleOptionsChange}
                        onOptimizeConfigChange={handleOptimizeConfigChange}
                        onCompute={computeGraphForCard}
                        onSelectGraph={handleSelectGraph}
                        onCloseGraph={handleCloseGraph}
                    />

                    {#if activeGraph?.data.optimization}
                        <div class="optim-results">
                            <span
                                ><strong>Target:</strong> "{activeGraph.data.optimization.label_str}" @ {(
                                    activeGraph.data.optimization.label_prob * 100
                                ).toFixed(1)}%</span
                            >
                            <span><strong>L0:</strong> {activeGraph.data.optimization.l0_total.toFixed(0)} active</span>
                        </div>
                    {/if}

                    {#if computeError}
                        <div class="error-banner">
                            {computeError}
                            <button onclick={() => computeGraphForCard()}>Retry</button>
                        </div>
                    {/if}

                    <div class="graph-area" class:loading={loadingCardId === activeCard.id}>
                        {#if loadingCardId === activeCard.id && loadingState}
                            <ComputeProgressOverlay state={loadingState} />
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
                                    onEdgeCountChange={(count) => (filteredEdgeCount = count)}
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
        background: var(--bg-base);
        padding: var(--space-3);
    }

    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 0;
        gap: var(--space-3);
    }

    .graph-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .card-tabs-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border-bottom: 1px solid var(--border-default);
    }

    .optim-results {
        display: flex;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border-bottom: 1px solid var(--border-subtle);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--accent-amber);
    }

    .optim-results strong {
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
    }

    .graph-area {
        flex: 1;
        position: relative;
        min-height: 400px;
    }

    .graph-area.loading {
        opacity: 0.5;
    }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
        text-align: center;
        padding: var(--space-4);
        font-family: var(--font-sans);
    }

    .empty-state.full {
        flex: 1;
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
    }

    .empty-state strong {
        color: var(--accent-amber);
    }

    .empty-state .hint {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .no-run-message {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: var(--text-muted);
        text-align: center;
        padding: var(--space-4);
        font-family: var(--font-sans);
    }

    .no-run-message .server-error {
        color: var(--status-negative-bright);
        margin-top: var(--space-2);
        font-family: var(--font-mono);
    }

    .warning-banner {
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--accent-amber-dim);
        border-left: 3px solid var(--accent-amber);
        color: var(--accent-amber);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .error-banner {
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--status-negative);
        border-left: 3px solid var(--status-negative-bright);
        display: flex;
        align-items: center;
        gap: var(--space-3);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--status-negative-bright);
    }

    .error-banner button {
        margin-left: auto;
        padding: var(--space-1) var(--space-2);
        background: var(--status-negative);
        color: var(--text-primary);
        border: none;
        cursor: pointer;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .error-banner button:hover {
        background: var(--status-negative-bright);
    }
</style>
