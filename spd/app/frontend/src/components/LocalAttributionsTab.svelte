<script lang="ts">
    import type { SvelteSet } from "svelte/reactivity";
    import * as mainApi from "../lib/api";
    import * as attrApi from "../lib/localAttributionsApi";
    import {
        filterInterventableNodes,
        type ActivationContextsSummary,
        type ComponentDetail,
        type GraphData,
        type PinnedNode,
        type PromptPreview,
        type TokenInfo,
    } from "../lib/localAttributionsTypes";
    import ComputeProgressOverlay from "./local-attr/ComputeProgressOverlay.svelte";
    import InterventionsView from "./local-attr/InterventionsView.svelte";
    import PromptCardHeader from "./local-attr/PromptCardHeader.svelte";
    import PromptCardTabs from "./local-attr/PromptCardTabs.svelte";
    import PromptPicker from "./local-attr/PromptPicker.svelte";
    import StagedNodesPanel from "./local-attr/StagedNodesPanel.svelte";
    import {
        defaultOptimizeConfig,
        type StoredGraph,
        type LoadingState,
        type OptimizeConfig,
        type PromptCard,
        type ViewSettings,
    } from "./local-attr/types";
    import ViewControls from "./local-attr/ViewControls.svelte";
    import LocalAttributionsGraph from "./LocalAttributionsGraph.svelte";

    /** Format token for display: strip leading space, add ## prefix if no leading space */
    function formatTokenDisplay(tokenString: string): string {
        if (tokenString.startsWith(" ")) {
            return tokenString.slice(1);
        }
        return "##" + tokenString;
    }

    // Props - activation contexts state is lifted to App.svelte
    type Props = {
        activationContextsSummary: ActivationContextsSummary | null;
        activationContextsMissing: boolean;
    };

    let { activationContextsSummary, activationContextsMissing }: Props = $props();

    // Server state
    let loadedRun = $state<mainApi.LoadedRun | null>(null);
    let serverError = $state<string | null>(null);

    // Available prompts (for picker)
    let prompts = $state<PromptPreview[]>([]);

    // All tokens for dropdown search
    let allTokens = $state<TokenInfo[]>([]);

    // Prompt cards state
    let promptCards = $state<PromptCard[]>([]);
    let activeCardId = $state<string | null>(null);

    // Prompt picker state
    let showPromptPicker = $state(false);
    let filteredPrompts = $state<PromptPreview[]>([]);
    let filterLoading = $state(false);
    let isAddingCustomPrompt = $state(false);

    // Loading state
    let loadingCardId = $state<string | null>(null);
    let loadingState = $state<LoadingState | null>(null);
    let computeError = $state<string | null>(null);

    // Intervention loading state
    let runningIntervention = $state(false);

    // Graph generation state
    let generatingGraphs = $state(false);
    let generateProgress = $state(0);
    let generateCount = $state(0);

    // Refetching state (for CI threshold/normalize changes) - tracks which graph is being refetched
    let refetchingGraphId = $state<string | null>(null);

    // Default view settings for new graphs
    const defaultViewSettings: ViewSettings = {
        topK: 200,
        componentGap: 4,
        layerGap: 30,
        normalizeEdges: "layer",
        ciThreshold: 0,
    };

    // Edge count is derived from the graph rendering, not stored per-graph
    let filteredEdgeCount = $state<number | null>(null);

    // Component details cache (shared across graphs)
    let componentDetailsCache = $state<Record<string, ComponentDetail>>({});
    let componentDetailsLoading = $state<Record<string, boolean>>({});

    // Pinned nodes for attributions graph
    let pinnedNodes = $state<PinnedNode[]>([]);

    async function loadComponentDetail(layer: string, cIdx: number) {
        const cacheKey = `${layer}:${cIdx}`;
        if (componentDetailsCache[cacheKey] || componentDetailsLoading[cacheKey]) return;

        componentDetailsLoading[cacheKey] = true;
        try {
            const detail = await attrApi.getComponentDetail(layer, cIdx);
            componentDetailsCache[cacheKey] = detail;
        } finally {
            componentDetailsLoading[cacheKey] = false;
        }
    }

    function handlePinnedNodesChange(nodes: PinnedNode[]) {
        pinnedNodes = nodes;
        // Load component details for any newly pinned nodes
        for (const node of nodes) {
            if (node.layer !== "wte" && node.layer !== "output") {
                loadComponentDetail(node.layer, node.cIdx);
            }
        }
    }

    // NOTE: Token selection is handled entirely by TokenDropdown, which provides the exact
    // token ID. We don't re-tokenize text because the same string (e.g. "art") can map to
    // different tokens depending on context (continuation "##art" vs word-initial " art").
    // The dropdown's onSelect callback sets labelTokenId directly.
    //
    // FUTURE: formatTokenDisplay() is WordPiece-specific. For BPE tokenizers (GPT-2 style),
    // the display logic will need to be tokenizer-aware.

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
            loadAllTokens();
            promptCards = [];
            activeCardId = null;
        } else if (currentRunId === null && previousRunId !== null) {
            previousRunId = null;
            prompts = [];
            allTokens = [];
            promptCards = [];
            activeCardId = null;
        }
    });

    async function loadServerStatus() {
        loadedRun = await mainApi.getStatus();
        serverError = null;
    }

    async function loadPromptsList() {
        prompts = await attrApi.listPrompts();
    }

    async function loadAllTokens() {
        allTokens = await attrApi.getAllTokens();
    }

    async function addPromptCard(promptId: number, tokens: string[], tokenIds: number[], isCustom: boolean) {
        const cardId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

        // Fetch stored graphs for this prompt (includes composer selection and intervention runs)
        let graphs: StoredGraph[] = [];
        const storedGraphs = await attrApi.getGraphs(
            promptId,
            defaultViewSettings.normalizeEdges,
            defaultViewSettings.ciThreshold,
        );
        graphs = await Promise.all(
            storedGraphs.map(async (data, idx) => {
                const isOptimized = !!data.optimization;
                const label = isOptimized ? `Optimized (${data.optimization!.steps} steps)` : "Standard";

                // Load intervention runs for this graph
                const runs = await mainApi.getInterventionRuns(data.id);

                return {
                    id: `graph-${idx}-${Date.now()}`,
                    dbId: data.id,
                    label,
                    data,
                    viewSettings: { ...defaultViewSettings },
                    composerSelection: filterInterventableNodes(Object.keys(data.nodeCiVals)),
                    interventionRuns: runs,
                    activeRunId: null,
                };
            }),
        );

        const newCard: PromptCard = {
            id: cardId,
            promptId,
            tokens,
            tokenIds,
            isCustom,
            graphs,
            activeGraphId: graphs.length > 0 ? graphs[0].id : null,
            activeView: "graph",
            newGraphConfig: defaultOptimizeConfig(),
            useOptimized: false,
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

    function handleUseOptimizedChange(useOptimized: boolean) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => (card.id === activeCard.id ? { ...card, useOptimized } : card));
    }

    function handleOptimizeConfigChange(partial: Partial<OptimizeConfig>) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) =>
            card.id === activeCard.id ? { ...card, newGraphConfig: { ...card.newGraphConfig, ...partial } } : card,
        );
    }

    function handleEnterNewGraphMode() {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => (card.id === activeCard.id ? { ...card, activeGraphId: null } : card));
    }

    // Switch between graph and interventions view
    function handleViewChange(view: "graph" | "interventions") {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => (card.id === activeCard.id ? { ...card, activeView: view } : card));
    }

    // Update composer selection for the active graph
    function handleComposerSelectionChange(selection: SvelteSet<string>) {
        if (!activeCard || !activeGraph) return;

        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) =>
                    g.id === activeGraph.id ? { ...g, composerSelection: selection, activeRunId: null } : g,
                ),
            };
        });
    }

    // Run intervention and save to DB
    async function handleRunIntervention() {
        if (!activeCard || !activeGraph || activeGraph.composerSelection.size === 0) return;

        runningIntervention = true;
        try {
            const text = activeCard.tokens.join("");
            const selectedNodes = Array.from(activeGraph.composerSelection);

            const run = await mainApi.runAndSaveIntervention({
                graph_id: activeGraph.dbId,
                text,
                selected_nodes: selectedNodes,
            });

            // Add run to local state and select it
            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: card.graphs.map((g) =>
                        g.id === activeGraph.id
                            ? { ...g, interventionRuns: [...g.interventionRuns, run], activeRunId: run.id }
                            : g,
                    ),
                };
            });
        } finally {
            runningIntervention = false;
        }
    }

    // Select a run and restore its selection state
    function handleSelectRun(runId: number) {
        if (!activeCard || !activeGraph) return;

        const run = activeGraph.interventionRuns.find((r) => r.id === runId);
        if (!run) return;

        // Restore selection from the run
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) =>
                    g.id === activeGraph.id
                        ? { ...g, composerSelection: filterInterventableNodes(run.selected_nodes), activeRunId: runId }
                        : g,
                ),
            };
        });
    }

    // Delete an intervention run
    async function handleDeleteRun(runId: number) {
        if (!activeCard || !activeGraph) return;

        await mainApi.deleteInterventionRun(runId);

        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    const newRuns = g.interventionRuns.filter((r) => r.id !== runId);
                    return {
                        ...g,
                        interventionRuns: newRuns,
                        activeRunId: g.activeRunId === runId ? null : g.activeRunId,
                    };
                }),
            };
        });
    }

    async function computeGraphForCard() {
        if (!activeCard || !activeCard.tokenIds || loadingCardId) return;

        loadingCardId = activeCard.id;
        computeError = null;

        const optConfig = activeCard.newGraphConfig;
        const isOptimized = activeCard.useOptimized;

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

        let data: GraphData;

        if (isOptimized) {
            const useCE = optConfig.ceLossCoeff > 0 && optConfig.labelTokenId !== null;
            const useKL = optConfig.klLossCoeff > 0;

            // Validate: at least one loss type must be active
            if (!useCE && !useKL) {
                throw new Error("At least one loss type must be active (set coeff > 0)");
            }
            // Validate: CE coeff > 0 requires label token
            if (optConfig.ceLossCoeff > 0 && !optConfig.labelTokenId) {
                throw new Error("Label token required when ce_coeff > 0");
            }

            // Build params with optional CE/KL settings
            const params: attrApi.ComputeGraphOptimizedParams = {
                promptId: activeCard.promptId,
                normalize: defaultViewSettings.normalizeEdges,
                impMinCoeff: optConfig.impMinCoeff,
                steps: optConfig.steps,
                pnorm: optConfig.pnorm,
                outputProbThreshold: 0.01,
                ciThreshold: defaultViewSettings.ciThreshold,
            };
            if (useCE) {
                params.labelToken = optConfig.labelTokenId!;
                params.ceLossCoeff = optConfig.ceLossCoeff;
            }
            if (useKL) {
                params.klLossCoeff = optConfig.klLossCoeff;
            }

            data = await attrApi.computeGraphOptimizedStreaming(params, (progress) => {
                if (!loadingState) return;
                if (progress.stage === "graph") {
                    loadingState.currentStage = 1;
                    loadingState.stages[1].progress = progress.current / progress.total;
                } else {
                    loadingState.stages[0].progress = progress.current / progress.total;
                }
            });
        } else {
            const params: attrApi.ComputeGraphParams = {
                promptId: activeCard.promptId,
                normalize: defaultViewSettings.normalizeEdges,
                ciThreshold: defaultViewSettings.ciThreshold,
            };
            data = await attrApi.computeGraphStreaming(params, (progress) => {
                if (!loadingState) return;
                loadingState.stages[0].progress = progress.current / progress.total;
            });
        }

        const graphId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        const label = isOptimized ? `Optimized (${optConfig.steps} steps)` : "Standard";

        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: [
                    ...card.graphs,
                    {
                        id: graphId,
                        dbId: data.id,
                        label,
                        data,
                        viewSettings: { ...defaultViewSettings },
                        composerSelection: filterInterventableNodes(Object.keys(data.nodeCiVals)),
                        interventionRuns: [],
                        activeRunId: null,
                    },
                ],
                activeGraphId: graphId,
            };
        });
        loadingCardId = null;
        loadingState = null;
    }

    // Refetch graph data when normalize or ciThreshold changes (these affect server-side filtering)
    async function refetchActiveGraphData() {
        if (!activeCard || !activeGraph) return;

        const { normalizeEdges, ciThreshold } = activeGraph.viewSettings;
        refetchingGraphId = activeGraph.id;
        try {
            const storedGraphs = await attrApi.getGraphs(activeCard.promptId, normalizeEdges, ciThreshold);
            const matchingData = storedGraphs.find((g) => g.id === activeGraph.dbId);

            if (!matchingData) {
                throw new Error("Could not find matching graph data after refetch");
            }

            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: card.graphs.map((g) => {
                        if (g.id !== activeGraph.id) return g;
                        return {
                            ...g,
                            data: matchingData,
                            composerSelection: filterInterventableNodes(Object.keys(matchingData.nodeCiVals)),
                        };
                    }),
                };
            });
        } finally {
            refetchingGraphId = null;
        }
    }

    function updateActiveGraphViewSettings(partial: Partial<ViewSettings>) {
        if (!activeCard || !activeGraph) return;

        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    return {
                        ...g,
                        viewSettings: { ...g.viewSettings, ...partial },
                    };
                }),
            };
        });
    }

    async function handleNormalizeChange(value: attrApi.NormalizeType) {
        updateActiveGraphViewSettings({ normalizeEdges: value });
        await refetchActiveGraphData();
    }

    async function handleCiThresholdChange(value: number) {
        updateActiveGraphViewSettings({ ciThreshold: value });
        await refetchActiveGraphData();
    }

    function handleTopKChange(value: number) {
        updateActiveGraphViewSettings({ topK: value });
    }

    function handleComponentGapChange(value: number) {
        updateActiveGraphViewSettings({ componentGap: value });
    }

    function handleLayerGapChange(value: number) {
        updateActiveGraphViewSettings({ layerGap: value });
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
                        stagedNodes={pinnedNodes}
                        filterByStaged={false}
                        {filterLoading}
                        {generatingGraphs}
                        {generateProgress}
                        {generateCount}
                        {isAddingCustomPrompt}
                        show={showPromptPicker}
                        onSelectPrompt={handleSelectPrompt}
                        onAddCustom={handleAddCustomPrompt}
                        onFilterToggle={() => {}}
                        onGenerate={handleGeneratePrompts}
                        onClose={() => (showPromptPicker = false)}
                    />
                </div>

                <div class="card-content">
                    {#if activeCard}
                        <!-- Prompt header with compute options and graph tabs -->
                        <PromptCardHeader
                            card={activeCard}
                            isLoading={loadingCardId === activeCard.id}
                            tokens={allTokens}
                            onUseOptimizedChange={handleUseOptimizedChange}
                            onOptimizeConfigChange={handleOptimizeConfigChange}
                            onCompute={computeGraphForCard}
                            onSelectGraph={handleSelectGraph}
                            onCloseGraph={handleCloseGraph}
                            onNewGraph={handleEnterNewGraphMode}
                        />

                        {#if activeGraph}
                            <!-- Sub-tabs within the selected graph -->
                            <div class="graph-view-tabs">
                                <button
                                    class="graph-view-tab"
                                    class:active={activeCard.activeView === "graph"}
                                    onclick={() => handleViewChange("graph")}
                                >
                                    Attributions
                                </button>
                                <button
                                    class="graph-view-tab"
                                    class:active={activeCard.activeView === "interventions"}
                                    onclick={() => handleViewChange("interventions")}
                                >
                                    Interventions
                                    {#if activeGraph.interventionRuns.length > 0}
                                        <span class="badge">{activeGraph.interventionRuns.length}</span>
                                    {/if}
                                </button>
                            </div>

                            {#if activeCard.activeView === "graph"}
                                <div class="graph-stats">
                                    {#if activeGraph.data.optimization}
                                        {#if activeGraph.data.optimization.label_str !== null && activeGraph.data.optimization.label_prob !== null}
                                            <span
                                                ><strong>Target:</strong> "{formatTokenDisplay(
                                                    activeGraph.data.optimization.label_str,
                                                )}"
                                                <span class="token-id"
                                                    >(#{activeGraph.data.optimization.label_token})</span
                                                >
                                                @ {(activeGraph.data.optimization.label_prob * 100).toFixed(1)}%</span
                                            >
                                        {/if}
                                        {#if activeGraph.data.optimization.kl_loss_coeff !== null}
                                            <span
                                                ><strong>KL Loss:</strong> coeff={activeGraph.data.optimization
                                                    .kl_loss_coeff}</span
                                            >
                                        {/if}
                                    {/if}
                                    <span
                                        ><strong>L0:</strong>
                                        {activeGraph.data.l0_total.toFixed(0)} active at ci threshold {activeGraph
                                            .viewSettings.ciThreshold}</span
                                    >
                                </div>

                                <!-- {#if computeError}
                                    <div class="error-banner">
                                        {computeError}
                                        <button onclick={() => computeGraphForCard()}>Retry</button>
                                    </div>
                                {/if} -->

                                <div class="graph-area" class:loading={loadingCardId === activeCard.id}>
                                    {#if loadingCardId === activeCard.id && loadingState}
                                        <ComputeProgressOverlay state={loadingState} />
                                    {/if}

                                    <ViewControls
                                        topK={activeGraph.viewSettings.topK}
                                        componentGap={activeGraph.viewSettings.componentGap}
                                        layerGap={activeGraph.viewSettings.layerGap}
                                        {filteredEdgeCount}
                                        normalizeEdges={activeGraph.viewSettings.normalizeEdges}
                                        ciThreshold={activeGraph.viewSettings.ciThreshold}
                                        ciThresholdLoading={refetchingGraphId === activeGraph.id}
                                        onTopKChange={handleTopKChange}
                                        onComponentGapChange={handleComponentGapChange}
                                        onLayerGapChange={handleLayerGapChange}
                                        onNormalizeChange={handleNormalizeChange}
                                        onCiThresholdChange={handleCiThresholdChange}
                                    />
                                    {#key activeGraph.id}
                                        <LocalAttributionsGraph
                                            data={activeGraph.data}
                                            topK={activeGraph.viewSettings.topK}
                                            componentGap={activeGraph.viewSettings.componentGap}
                                            layerGap={activeGraph.viewSettings.layerGap}
                                            {activationContextsSummary}
                                            stagedNodes={pinnedNodes}
                                            {componentDetailsCache}
                                            {componentDetailsLoading}
                                            onStagedNodesChange={handlePinnedNodesChange}
                                            onLoadComponentDetail={loadComponentDetail}
                                            onEdgeCountChange={(count) => (filteredEdgeCount = count)}
                                        />
                                    {/key}
                                </div>
                                <StagedNodesPanel
                                    stagedNodes={pinnedNodes}
                                    {componentDetailsCache}
                                    {componentDetailsLoading}
                                    {activationContextsSummary}
                                    outputProbs={activeGraph.data.outputProbs}
                                    tokens={activeCard.tokens}
                                    onStagedNodesChange={handlePinnedNodesChange}
                                />
                            {:else}
                                <!-- Interventions view -->
                                <InterventionsView
                                    graph={activeGraph}
                                    tokens={activeCard.tokens}
                                    initialTopK={activeGraph.viewSettings.topK}
                                    {activationContextsSummary}
                                    {componentDetailsCache}
                                    {componentDetailsLoading}
                                    {runningIntervention}
                                    onLoadComponentDetail={loadComponentDetail}
                                    onSelectionChange={handleComposerSelectionChange}
                                    onRunIntervention={handleRunIntervention}
                                    onSelectRun={handleSelectRun}
                                    onDeleteRun={handleDeleteRun}
                                />
                            {/if}
                        {:else}
                            <!-- No graph yet -->
                            {#if computeError}
                                <div class="error-banner">
                                    {computeError}
                                    <button onclick={() => computeGraphForCard()}>Retry</button>
                                </div>
                            {/if}

                            <div class="graph-area" class:loading={loadingCardId === activeCard.id}>
                                {#if loadingCardId === activeCard.id && loadingState}
                                    <ComputeProgressOverlay state={loadingState} />
                                {:else}
                                    <div class="empty-state">
                                        <p>Click <strong>Compute</strong> to generate the attribution graph</p>
                                    </div>
                                {/if}
                            </div>
                        {/if}
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

    .graph-view-tabs {
        display: flex;
        gap: var(--space-1);
        margin-bottom: var(--space-2);
        padding-top: var(--space-2);
        border-top: 1px solid var(--border-subtle);
    }

    .graph-view-tab {
        padding: var(--space-1) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
        display: inline-flex;
        align-items: center;
        gap: var(--space-1);
    }

    .graph-view-tab:hover {
        color: var(--text-primary);
        border-color: var(--border-strong);
        background: var(--bg-surface);
    }

    .graph-view-tab.active {
        color: white;
        background: var(--accent-primary);
        border-color: var(--accent-primary);
    }

    .graph-view-tab .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 16px;
        height: 16px;
        padding: 0 4px;
        font-size: var(--text-xs);
        font-weight: 600;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }

    .graph-view-tab.active .badge {
        background: rgba(255, 255, 255, 0.3);
    }

    .graph-stats {
        display: flex;
        gap: var(--space-4);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--accent-primary);
    }

    .graph-stats strong {
        color: var(--text-muted);
        font-weight: 500;
    }

    .graph-stats .token-id {
        color: var(--text-muted);
        font-size: var(--text-xs);
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
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
        text-align: center;
        padding: var(--space-4);
        font-family: var(--font-sans);
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
