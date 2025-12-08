<script lang="ts">
    import * as mainApi from "../lib/api";
    import * as attrApi from "../lib/localAttributionsApi";
    import type {
        ActivationContextsSummary,
        ComponentDetail,
        GraphData,
        PinnedNode,
        PromptPreview,
    } from "../lib/localAttributionsTypes";
    import PinnedComponentsPanel from "./local-attr/PinnedComponentsPanel.svelte";
    import ComputeProgressOverlay from "./local-attr/ComputeProgressOverlay.svelte";
    import PromptCardHeader from "./local-attr/PromptCardHeader.svelte";
    import PromptCardTabs from "./local-attr/PromptCardTabs.svelte";
    import PromptPicker from "./local-attr/PromptPicker.svelte";
    import type { StoredGraph, ComputeOptions, LoadingState, OptimizeConfig, PromptCard } from "./local-attr/types";
    import ViewControls from "./local-attr/ViewControls.svelte";
    import LocalAttributionsGraph from "./LocalAttributionsGraph.svelte";

    // Props - pinnedNodes are managed in App.svelte (shared with InterventionTab)
    type Props = {
        pinnedNodes: PinnedNode[];
        onPinnedNodesChange: (nodes: PinnedNode[]) => void;
        onGoToIntervention: (text: string) => void;
    };

    let { pinnedNodes, onPinnedNodesChange, onGoToIntervention }: Props = $props();

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
    let filterByPinned = $state(false);
    let filteredPrompts = $state<PromptPreview[]>([]);
    let filterLoading = $state(false);
    let isAddingCustomPrompt = $state(false);

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
    let topK = $state(200);
    let nodeLayout = $state<"importance" | "shuffled" | "jittered">("importance");
    let componentGap = $state(4);
    let layerGap = $state(30);
    let filteredEdgeCount = $state<number | null>(null);
    let normalizeEdges = $state<attrApi.NormalizeType>("layer");

    // Compute options
    let computeOptions = $state<ComputeOptions>({
        maxMeanCI: 1.0,
        ciThreshold: 1e-6,
        normalizeEdges: "layer", // kept for compute, but view uses normalizeEdges state
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

    // Component details cache (shared across graphs, persists across graph switches)
    let componentDetailsCache = $state<Record<string, ComponentDetail>>({});
    let componentDetailsLoading = $state<Record<string, boolean>>({});

    async function loadComponentDetail(layer: string, cIdx: number) {
        const cacheKey = `${layer}:${cIdx}`;
        if (componentDetailsCache[cacheKey] || componentDetailsLoading[cacheKey]) return;

        componentDetailsLoading[cacheKey] = true;
        try {
            const detail = await attrApi.getComponentDetail(layer, cIdx);
            componentDetailsCache[cacheKey] = detail;
        } catch (e) {
            console.error(`Failed to load component detail for ${cacheKey}:`, e);
        } finally {
            componentDetailsLoading[cacheKey] = false;
        }
    }

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

    async function addPromptCard(promptId: number, tokens: string[], tokenIds: number[], isCustom: boolean) {
        const cardId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

        // Fetch stored graphs for this prompt
        let graphs: StoredGraph[] = [];
        try {
            const storedGraphs = await attrApi.getGraphs(promptId, normalizeEdges, computeOptions.ciThreshold);
            graphs = storedGraphs.map((data, idx) => {
                const isOptimized = !!data.optimization;
                const label = isOptimized
                    ? `Optimized (${data.optimization!.steps} steps)`
                    : "Standard";
                return {
                    id: `graph-${idx}-${Date.now()}`,
                    label,
                    data,
                };
            });
        } catch (e) {
            console.warn("Failed to fetch graphs:", e);
        }

        const newCard: PromptCard = {
            id: cardId,
            promptId,
            tokens,
            tokenIds,
            isCustom,
            graphs,
            activeGraphId: graphs.length > 0 ? graphs[0].id : null,
        };
        promptCards = [...promptCards, newCard];
        activeCardId = cardId;
    }

    function handleSelectPrompt(prompt: PromptPreview) {
        addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, false);
    }

    async function handleAddCustomPrompt(text: string) {
        isAddingCustomPrompt = true;
        try {
            const prompt = await attrApi.createCustomPrompt(text);
            await addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, true);
        } finally {
            isAddingCustomPrompt = false;
        }
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
        const oldCiThreshold = computeOptions.ciThreshold;
        computeOptions = { ...computeOptions, ...partial };

        // If ciThreshold changed, re-fetch all graphs with new threshold
        if (partial.ciThreshold !== undefined && partial.ciThreshold !== oldCiThreshold) {
            handleCiThresholdChange(partial.ciThreshold);
        }
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
                        promptId: activeCard.promptId,
                        labelToken: optConfig.labelTokenId,
                        normalize: computeOptions.normalizeEdges,
                        impMinCoeff: optConfig.impMinCoeff,
                        ceLossCoeff: optConfig.ceLossCoeff,
                        steps: optConfig.steps,
                        pnorm: optConfig.pnorm,
                        outputProbThreshold: 0.01,
                        ciThreshold: computeOptions.ciThreshold,
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
                        promptId: activeCard.promptId,
                        normalize: computeOptions.normalizeEdges,
                        ciThreshold: computeOptions.ciThreshold,
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

    function handlePinnedNodesChangeWithFilter(nodes: PinnedNode[]) {
        onPinnedNodesChange(nodes);
        if (filterByPinned) {
            filterPromptsByPinned();
        }
    }

    async function refetchAllGraphs() {
        // Re-fetch all graphs for all cards with current settings
        const updatedCards = await Promise.all(
            promptCards.map(async (card) => {
                if (card.graphs.length === 0) return card;

                try {
                    const storedGraphs = await attrApi.getGraphs(card.promptId, normalizeEdges, computeOptions.ciThreshold);
                    const graphs = storedGraphs.map((data, idx) => {
                        const isOptimized = !!data.optimization;
                        const label = isOptimized
                            ? `Optimized (${data.optimization!.steps} steps)`
                            : "Standard";
                        return {
                            id: `graph-${idx}-${Date.now()}`,
                            label,
                            data,
                        };
                    });
                    return {
                        ...card,
                        graphs,
                        activeGraphId: graphs.length > 0 ? graphs[0].id : null,
                    };
                } catch (e) {
                    console.warn("Failed to re-fetch graphs for card:", card.id, e);
                    return card;
                }
            }),
        );
        promptCards = updatedCards;
    }

    async function handleNormalizeChange(value: attrApi.NormalizeType) {
        normalizeEdges = value;
        // Also sync to compute options so new computes use the same normalization
        computeOptions.normalizeEdges = value;
        await refetchAllGraphs();
    }

    async function handleCiThresholdChange(value: number) {
        computeOptions.ciThreshold = value;
        await refetchAllGraphs();
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

            <div class="graph-container">
                <div class="card-tabs-row">
                    <PromptCardTabs
                        cards={promptCards}
                        {activeCardId}
                        onSelectCard={(id) => (activeCardId = id)}
                        onCloseCard={handleCloseCard}
                        onAddClick={() => (showPromptPicker = !showPromptPicker)}
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
                        {isAddingCustomPrompt}
                        show={showPromptPicker}
                        onSelectPrompt={handleSelectPrompt}
                        onAddCustom={handleAddCustomPrompt}
                        onFilterToggle={handleFilterToggle}
                        onGenerate={handleGeneratePrompts}
                        onClose={() => (showPromptPicker = false)}
                    />
                </div>

                <div class="card-content">
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
                                <span
                                    ><strong>L0:</strong>
                                    {activeGraph.data.optimization.l0_total.toFixed(0)} active</span
                                >
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
                                <ViewControls
                                    {topK}
                                    {nodeLayout}
                                    {componentGap}
                                    {layerGap}
                                    {filteredEdgeCount}
                                    {normalizeEdges}
                                    onTopKChange={(v) => (topK = v)}
                                    onLayoutChange={(v) => (nodeLayout = v)}
                                    onComponentGapChange={(v) => (componentGap = v)}
                                    onLayerGapChange={(v) => (layerGap = v)}
                                    onNormalizeChange={handleNormalizeChange}
                                />
                                {#key activeGraph.id}
                                    <LocalAttributionsGraph
                                        data={activeGraph.data}
                                        {topK}
                                        {nodeLayout}
                                        {componentGap}
                                        {layerGap}
                                        {activationContextsSummary}
                                        {pinnedNodes}
                                        {componentDetailsCache}
                                        {componentDetailsLoading}
                                        onPinnedNodesChange={handlePinnedNodesChangeWithFilter}
                                        onLoadComponentDetail={loadComponentDetail}
                                        onEdgeCountChange={(count) => (filteredEdgeCount = count)}
                                    />
                                {/key}
                                <PinnedComponentsPanel
                                    {pinnedNodes}
                                    {componentDetailsCache}
                                    outputProbs={activeGraph.data.outputProbs}
                                    tokens={activeGraph.data.tokens}
                                    onPinnedNodesChange={handlePinnedNodesChangeWithFilter}
                                    {onGoToIntervention}
                                />
                            {:else if !loadingCardId}
                                <div class="empty-state">
                                    <p>Click <strong>Compute</strong> to generate the attribution graph</p>
                                </div>
                            {/if}
                        </div>
                    {:else}
                        <div class="empty-state">
                            <p>Click <strong>+ Add Prompt</strong> to get started</p>
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
        background: var(--bg-base);
    }

    .main-content {
        flex: 1;
        gap: var(--space-4);
        display: flex;
        flex-direction: column;
        min-width: 0;
        padding: var(--space-6);
    }

    .graph-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .card-tabs-row {
        display: flex;
        align-items: center;
        margin-bottom: var(--space-2);
        background: var(--bg-elevated);
        position: relative;
    }

    .card-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        min-height: 0;
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
    }

    .optim-results {
        display: flex;
        gap: var(--space-4);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--accent-primary);
    }

    .optim-results strong {
        color: var(--text-muted);
        font-weight: 500;
    }

    .graph-area {
        flex: 1;
        position: relative;
        min-height: 400px;
        /* border: 1px solid var(--border-default); */
    }

    .graph-area.loading {
        opacity: 0.5;
    }

    .empty-state {
        display: flex;
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
        text-align: center;
        padding: var(--space-4);
        font-family: var(--font-sans);
        background: var(--bg-surface);
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
    }

    .empty-state strong {
        color: var(--accent-primary);
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
        border: 1px solid var(--accent-primary-dim);
        border-radius: var(--radius-sm);
        color: var(--accent-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .error-banner {
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--status-negative);
        border-radius: var(--radius-sm);
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
        color: white;
        border: none;
    }

    .error-banner button:hover {
        background: var(--status-negative-bright);
    }
</style>
