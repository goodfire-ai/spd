<script lang="ts">
    import { getContext } from "svelte";
    import * as api from "../lib/api";
    import {
        filterInterventableNodes,
        type GraphData,
        type PinnedNode,
        type PromptPreview,
        type TokenInfo,
    } from "../lib/promptAttributionsTypes";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ComputeProgressOverlay from "./prompt-attr/ComputeProgressOverlay.svelte";
    import InterventionsView from "./prompt-attr/InterventionsView.svelte";
    import PromptCardHeader from "./prompt-attr/PromptCardHeader.svelte";
    import PromptCardTabs from "./prompt-attr/PromptCardTabs.svelte";
    import PromptPicker from "./prompt-attr/PromptPicker.svelte";
    import StagedNodesPanel from "./prompt-attr/StagedNodesPanel.svelte";
    import {
        defaultOptimizeConfig,
        type ActionState,
        type ComposerState,
        type StoredGraph,
        type GraphComputeState,
        type PromptGenerateState,
        type OptimizeConfig,
        type PromptCard,
        type ViewSettings,
    } from "./prompt-attr/types";
    import ViewControls from "./prompt-attr/ViewControls.svelte";
    import PromptAttributionsGraph from "./PromptAttributionsGraph.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    /** Generate a display label for a graph based on its type */
    function getGraphLabel(data: GraphData): string {
        switch (data.graphType) {
            case "standard":
                return "Standard";
            case "optimized":
                return data.optimization ? `Optimized (${data.optimization.steps} steps)` : "Optimized";
            case "manual":
                return `Manual (${Object.keys(data.nodeCiVals).length} nodes)`;
        }
    }

    type Props = {
        prompts: PromptPreview[];
        allTokens: TokenInfo[];
    };

    let { prompts, allTokens }: Props = $props();

    // Prompt cards state
    let promptCards = $state<PromptCard[]>([]);
    let activeCardPromptId = $state<number | null>(null);

    // Prompt picker state
    let showPromptPicker = $state(false);
    let filteredPrompts = $state<PromptPreview[]>([]);
    let filterLoading = $state(false);
    let isAddingCustomPrompt = $state(false);
    let promptCardLoading = $state<ActionState>({ status: "idle" });

    // Graph computation state
    let graphCompute = $state<GraphComputeState>({ status: "idle" });

    // Intervention loading state
    let runningIntervention = $state(false);
    let generatingSubgraph = $state(false);

    // Prompt generation state
    let promptGenerate = $state<PromptGenerateState>({ status: "idle" });

    // Refetching state (for CI threshold/normalize changes) - tracks which graph is being refetched
    let refetchingGraphId = $state<number | null>(null);

    // Composer state - transient UI state for interventions, keyed by graph ID
    let composerStates = $state<Record<number, ComposerState>>({});

    // Helper to get or create composer state for a graph
    function getComposerState(graphId: number, nodeKeys?: Iterable<string>): ComposerState {
        if (!composerStates[graphId]) {
            composerStates[graphId] = {
                selection: nodeKeys ? new Set(filterInterventableNodes(nodeKeys)) : new Set(),
                activeRunId: null,
            };
        }
        return composerStates[graphId];
    }

    // Derived: composer state for the active graph
    const activeComposerState = $derived.by(() => {
        if (!activeGraph) return null;
        return composerStates[activeGraph.id] ?? null;
    });

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
    let hideUnpinnedEdges = $state(false);
    let hideNodeCard = $state(false);

    // Pinned nodes for attributions graph
    let pinnedNodes = $state<PinnedNode[]>([]);

    function handlePinnedNodesChange(nodes: PinnedNode[]) {
        pinnedNodes = nodes;
    }

    // NOTE: Token selection is handled entirely by TokenDropdown, which provides the exact
    // token ID. We don't re-tokenize text because the same string (e.g. "art") can map to
    // different tokens depending on context (continuation "##art" vs word-initial " art").
    // The dropdown's onSelect callback sets labelTokenId directly.

    // Derived state
    const activeCard = $derived(promptCards.find((c) => c.id === activeCardPromptId) ?? null);
    const activeGraph = $derived.by(() => {
        const card = promptCards.find((c) => c.id === activeCardPromptId);
        if (!card) return null;
        return card.graphs.find((g) => g.id === card.activeGraphId) ?? null;
    });

    async function addPromptCard(promptId: number, tokens: string[], tokenIds: number[], isCustom: boolean) {
        promptCardLoading = { status: "loading" };
        try {
            await addPromptCardInner(promptId, tokens, tokenIds, isCustom);
            promptCardLoading = { status: "idle" };
        } catch (error) {
            promptCardLoading = { status: "error", error: String(error) };
        }
    }

    async function addPromptCardInner(promptId: number, tokens: string[], tokenIds: number[], isCustom: boolean) {
        // Fetch stored graphs for this prompt
        const storedGraphs = await api.getGraphs(
            promptId,
            defaultViewSettings.normalizeEdges,
            defaultViewSettings.ciThreshold,
        );
        const graphs: StoredGraph[] = await Promise.all(
            storedGraphs.map(async (data) => {
                // Load intervention runs for this graph
                const runs = await api.getInterventionRuns(data.id);

                // Initialize composer state for this graph
                getComposerState(data.id, Object.keys(data.nodeCiVals));

                return {
                    id: data.id,
                    label: getGraphLabel(data),
                    data,
                    viewSettings: { ...defaultViewSettings },
                    interventionRuns: runs,
                };
            }),
        );

        const newCard: PromptCard = {
            id: promptId,
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
        activeCardPromptId = promptId;
    }

    function handleSelectPrompt(prompt: PromptPreview) {
        // If prompt is already open as a card, just focus it
        const existingCard = promptCards.find((c) => c.id === prompt.id);
        if (existingCard) {
            activeCardPromptId = prompt.id;
            return;
        }
        addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, false);
    }

    async function handleAddCustomPrompt(text: string) {
        isAddingCustomPrompt = true;
        try {
            const prompt = await api.createCustomPrompt(text);
            // If prompt already exists (returned existing ID), just focus it
            const existingCard = promptCards.find((c) => c.id === prompt.id);
            if (existingCard) {
                activeCardPromptId = prompt.id;
                return;
            }
            await addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, true);
        } finally {
            isAddingCustomPrompt = false;
        }
    }

    function handleCloseCard(cardId: number) {
        promptCards = promptCards.filter((c) => c.id !== cardId);
        if (activeCardPromptId === cardId) {
            activeCardPromptId = promptCards.length > 0 ? promptCards[promptCards.length - 1].id : null;
        }
    }

    function handleSelectGraph(graphId: number) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) =>
            card.id === activeCard.id ? { ...card, activeGraphId: graphId } : card,
        );
    }

    function handleCloseGraph(graphId: number) {
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
    function handleComposerSelectionChange(selection: Set<string>) {
        if (!activeGraph) return;
        composerStates[activeGraph.id] = {
            selection,
            activeRunId: null, // Clear active run when selection changes manually
        };
    }

    // Run intervention and save to DB
    async function handleRunIntervention() {
        const composerState = activeComposerState;
        if (!activeCard || !activeGraph || !composerState) return;

        runningIntervention = true;
        try {
            const text = activeCard.tokens.join("");
            const selectedNodes = Array.from(composerState.selection);

            const run = await api.runAndSaveIntervention({
                graph_id: activeGraph.id,
                text,
                selected_nodes: selectedNodes,
            });

            // Add run to local state
            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: card.graphs.map((g) =>
                        g.id === activeGraph.id ? { ...g, interventionRuns: [...g.interventionRuns, run] } : g,
                    ),
                };
            });

            // Select the new run
            composerStates[activeGraph.id] = { ...composerState, activeRunId: run.id };
        } finally {
            runningIntervention = false;
        }
    }

    // Select a run and restore its selection state
    function handleSelectRun(runId: number) {
        if (!activeGraph) return;

        const run = activeGraph.interventionRuns.find((r) => r.id === runId);
        if (!run) return;

        // Restore selection from the run
        composerStates[activeGraph.id] = {
            selection: new Set(filterInterventableNodes(run.selected_nodes)),
            activeRunId: runId,
        };
    }

    // Delete an intervention run
    async function handleDeleteRun(runId: number) {
        if (!activeCard || !activeGraph) return;

        await api.deleteInterventionRun(runId);

        // Remove from persisted state
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    return { ...g, interventionRuns: g.interventionRuns.filter((r) => r.id !== runId) };
                }),
            };
        });

        // Clear activeRunId if we deleted the active run
        const composerState = composerStates[activeGraph.id];
        if (composerState?.activeRunId === runId) {
            composerStates[activeGraph.id] = { ...composerState, activeRunId: null };
        }
    }

    // Fork an intervention run with modified tokens
    async function handleForkRun(runId: number, tokenReplacements: [number, number][]) {
        if (!activeCard || !activeGraph) throw new Error("No active card or graph");

        const forkedRun = await api.forkInterventionRun(runId, tokenReplacements);

        // Add forked run to the parent run's forked_runs list
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    return {
                        ...g,
                        interventionRuns: g.interventionRuns.map((r) => {
                            if (r.id !== runId) return r;
                            return {
                                ...r,
                                forked_runs: [...(r.forked_runs || []), forkedRun],
                            };
                        }),
                    };
                }),
            };
        });

        return forkedRun;
    }

    // Delete a forked intervention run
    async function handleDeleteFork(forkId: number) {
        if (!activeCard || !activeGraph) return;

        await api.deleteForkedInterventionRun(forkId);

        // Remove the forked run from state
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    return {
                        ...g,
                        interventionRuns: g.interventionRuns.map((r) => ({
                            ...r,
                            forked_runs: (r.forked_runs || []).filter((f) => f.id !== forkId),
                        })),
                    };
                }),
            };
        });
    }

    async function handleGenerateGraphFromSelection() {
        const composerState = activeComposerState;
        if (!activeCard || !activeGraph || !composerState) {
            return;
        }

        // Validate selection is not empty (defense in depth - button should be disabled)
        if (composerState.selection.size === 0) {
            console.warn("handleGenerateGraphFromSelection called with empty selection");
            return;
        }

        const cardId = activeCard.id;
        const includedNodes = Array.from(composerState.selection);

        generatingSubgraph = true;
        graphCompute = {
            status: "computing",
            cardId,
            progress: {
                stages: [{ name: "Computing graph from selection", progress: 0 }],
                currentStage: 0,
            },
        };

        try {
            const data = await api.computeGraphStreaming(
                {
                    promptId: cardId,
                    normalize: activeGraph.viewSettings.normalizeEdges,
                    ciThreshold: activeGraph.viewSettings.ciThreshold,
                    includedNodes,
                },
                (progress) => {
                    if (graphCompute.status === "computing") {
                        graphCompute.progress.stages[0].progress = progress.current / progress.total;
                    }
                },
            );

            // Initialize composer state for the new graph
            getComposerState(data.id, Object.keys(data.nodeCiVals));

            promptCards = promptCards.map((card) => {
                if (card.id !== cardId) return card;

                // Check if graph with this ID already exists (get-or-create semantics from backend)
                const existingGraph = card.graphs.find((g) => g.id === data.id);
                if (existingGraph) {
                    // Graph already exists, just select it
                    return {
                        ...card,
                        activeGraphId: data.id,
                        activeView: "graph",
                    };
                }

                // Add new graph
                return {
                    ...card,
                    graphs: [
                        ...card.graphs,
                        {
                            id: data.id,
                            label: getGraphLabel(data),
                            data,
                            viewSettings: { ...activeGraph.viewSettings },
                            interventionRuns: [],
                        },
                    ],
                    activeGraphId: data.id,
                    activeView: "graph",
                };
            });

            graphCompute = { status: "idle" };
        } catch (error) {
            graphCompute = { status: "idle" };
            alert(String(error));
        } finally {
            generatingSubgraph = false;
        }
    }

    async function computeGraphForCard() {
        if (!activeCard || !activeCard.tokenIds || graphCompute.status === "computing") return;

        const optConfig = activeCard.newGraphConfig;
        const isOptimized = activeCard.useOptimized;
        const cardId = activeCard.id;

        const initialProgress = isOptimized
            ? {
                  stages: [
                      { name: "Optimizing", progress: 0 },
                      { name: "Computing graph", progress: 0 },
                  ],
                  currentStage: 0,
              }
            : {
                  stages: [{ name: "Computing graph", progress: 0 }],
                  currentStage: 0,
              };

        graphCompute = { status: "computing", cardId, progress: initialProgress };

        try {
            let data: GraphData;

            if (isOptimized) {
                const useCE = optConfig.ceLossCoeff > 0 && optConfig.labelTokenId !== null;
                const useKL = optConfig.klLossCoeff > 0;
                // Validate: at least one loss type must be active
                if (!useCE && !useKL) {
                    throw new Error("At least one loss type must be active (set coeff > 0)");
                }
                // Validate: CE coeff > 0 requires label token
                if (optConfig.ceLossCoeff > 0 && optConfig.labelTokenId === null) {
                    throw new Error("Label token required when ce_coeff > 0");
                }

                // Build params with optional CE/KL settings
                const params: api.ComputeGraphOptimizedParams = {
                    promptId: cardId,
                    normalize: defaultViewSettings.normalizeEdges,
                    impMinCoeff: optConfig.impMinCoeff,
                    steps: optConfig.steps,
                    pnorm: optConfig.pnorm,
                    beta: optConfig.beta,
                    outputProbThreshold: 0.01,
                    ciThreshold: defaultViewSettings.ciThreshold,
                    maskType: optConfig.maskType,
                };
                if (useCE) {
                    params.labelToken = optConfig.labelTokenId!;
                    params.ceLossCoeff = optConfig.ceLossCoeff;
                }
                if (useKL) {
                    params.klLossCoeff = optConfig.klLossCoeff;
                }

                data = await api.computeGraphOptimizedStreaming(params, (progress) => {
                    if (graphCompute.status !== "computing") return;
                    if (progress.stage === "graph") {
                        graphCompute.progress.currentStage = 1;
                        graphCompute.progress.stages[1].progress = progress.current / progress.total;
                    } else {
                        graphCompute.progress.stages[0].progress = progress.current / progress.total;
                    }
                });
            } else {
                const params: api.ComputeGraphParams = {
                    promptId: cardId,
                    normalize: defaultViewSettings.normalizeEdges,
                    ciThreshold: defaultViewSettings.ciThreshold,
                };
                data = await api.computeGraphStreaming(params, (progress) => {
                    if (graphCompute.status !== "computing") return;
                    graphCompute.progress.stages[0].progress = progress.current / progress.total;
                });
            }

            // Initialize composer state for the new graph
            getComposerState(data.id, Object.keys(data.nodeCiVals));

            promptCards = promptCards.map((card) => {
                if (card.id !== cardId) return card;

                // Check if graph with this ID already exists (defensive check)
                const existingGraph = card.graphs.find((g) => g.id === data.id);
                if (existingGraph) {
                    return { ...card, activeGraphId: data.id };
                }

                return {
                    ...card,
                    graphs: [
                        ...card.graphs,
                        {
                            id: data.id,
                            label: getGraphLabel(data),
                            data,
                            viewSettings: { ...defaultViewSettings },
                            interventionRuns: [],
                        },
                    ],
                    activeGraphId: data.id,
                };
            });
            graphCompute = { status: "idle" };
        } catch (error) {
            graphCompute = { status: "error", error: String(error) };
        }
    }

    // Refetch graph data when normalize or ciThreshold changes (these affect server-side filtering)
    async function refetchActiveGraphData() {
        if (!activeCard || !activeGraph) return;

        const { normalizeEdges, ciThreshold } = activeGraph.viewSettings;
        refetchingGraphId = activeGraph.id;
        try {
            const storedGraphs = await api.getGraphs(activeCard.id, normalizeEdges, ciThreshold);
            const matchingData = storedGraphs.find((g) => g.id === activeGraph.id);

            if (!matchingData) {
                throw new Error("Could not find matching graph data after refetch");
            }

            // Update graph data
            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: card.graphs.map((g) => (g.id !== activeGraph.id ? g : { ...g, data: matchingData })),
                };
            });

            // Update composer selection with new node keys
            const composerState = composerStates[activeGraph.id];
            composerStates[activeGraph.id] = {
                selection: new Set(filterInterventableNodes(Object.keys(matchingData.nodeCiVals))),
                activeRunId: composerState?.activeRunId ?? null,
            };
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

    async function handleNormalizeChange(value: api.NormalizeType) {
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
        if (promptGenerate.status === "generating") return;

        promptGenerate = { status: "generating", progress: 0, count: 0 };

        try {
            await api.generatePrompts({ nPrompts }, (progress: number, count: number) => {
                if (promptGenerate.status === "generating") {
                    promptGenerate.progress = progress;
                    promptGenerate.count = count;
                }
            });
            await runState.refreshPrompts();
            promptGenerate = { status: "idle" };
        } catch (error) {
            promptGenerate = { status: "error", error: String(error) };
        }
    }
</script>

<div class="prompt-attributions-tab">
    <div class="main-content">
        <div class="graph-container">
            <div class="card-tabs-row">
                <PromptCardTabs
                    cards={promptCards}
                    activeCardId={activeCardPromptId}
                    onSelectCard={(id) => (activeCardPromptId = id)}
                    onCloseCard={handleCloseCard}
                    onAddClick={() => (showPromptPicker = !showPromptPicker)}
                />
                <PromptPicker
                    {prompts}
                    {filteredPrompts}
                    stagedNodes={pinnedNodes}
                    filterByStaged={false}
                    {filterLoading}
                    {promptGenerate}
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
                        isLoading={graphCompute.status === "computing" && graphCompute.cardId === activeCard.id}
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
                            <div class="graph-area">
                                <ViewControls
                                    topK={activeGraph.viewSettings.topK}
                                    componentGap={activeGraph.viewSettings.componentGap}
                                    layerGap={activeGraph.viewSettings.layerGap}
                                    {filteredEdgeCount}
                                    normalizeEdges={activeGraph.viewSettings.normalizeEdges}
                                    ciThreshold={refetchingGraphId === activeGraph.id
                                        ? { status: "loading" }
                                        : { status: "loaded", data: activeGraph.viewSettings.ciThreshold }}
                                    {hideUnpinnedEdges}
                                    {hideNodeCard}
                                    onTopKChange={handleTopKChange}
                                    onComponentGapChange={handleComponentGapChange}
                                    onLayerGapChange={handleLayerGapChange}
                                    onNormalizeChange={handleNormalizeChange}
                                    onCiThresholdChange={handleCiThresholdChange}
                                    onHideUnpinnedEdgesChange={(v) => (hideUnpinnedEdges = v)}
                                    onHideNodeCardChange={(v) => (hideNodeCard = v)}
                                />
                                <div class="graph-info">
                                    <span class="l0-info"
                                        ><strong>L0:</strong>
                                        {activeGraph.data.l0_total.toFixed(0)} active at ci threshold {activeGraph
                                            .viewSettings.ciThreshold}</span
                                    >
                                    {#if pinnedNodes.length > 0}
                                        <span class="pinned-count">{pinnedNodes.length} pinned</span>
                                    {/if}
                                </div>
                                {#key activeGraph.id}
                                    <PromptAttributionsGraph
                                        data={activeGraph.data}
                                        topK={activeGraph.viewSettings.topK}
                                        componentGap={activeGraph.viewSettings.componentGap}
                                        layerGap={activeGraph.viewSettings.layerGap}
                                        {hideUnpinnedEdges}
                                        {hideNodeCard}
                                        stagedNodes={pinnedNodes}
                                        onStagedNodesChange={handlePinnedNodesChange}
                                        onEdgeCountChange={(count) => (filteredEdgeCount = count)}
                                    />
                                {/key}
                            </div>
                            <StagedNodesPanel
                                stagedNodes={pinnedNodes}
                                outputProbs={activeGraph.data.outputProbs}
                                tokens={activeCard.tokens}
                                edgesBySource={activeGraph.data.edgesBySource}
                                edgesByTarget={activeGraph.data.edgesByTarget}
                                onStagedNodesChange={handlePinnedNodesChange}
                            />
                        {:else if activeComposerState}
                            <InterventionsView
                                graph={activeGraph}
                                composerSelection={activeComposerState.selection}
                                activeRunId={activeComposerState.activeRunId}
                                tokens={activeCard.tokens}
                                tokenIds={activeCard.tokenIds}
                                {allTokens}
                                topK={activeGraph.viewSettings.topK}
                                componentGap={activeGraph.viewSettings.componentGap}
                                layerGap={activeGraph.viewSettings.layerGap}
                                normalizeEdges={activeGraph.viewSettings.normalizeEdges}
                                ciThreshold={refetchingGraphId === activeGraph.id
                                    ? { status: "loading" }
                                    : { status: "loaded", data: activeGraph.viewSettings.ciThreshold }}
                                {hideUnpinnedEdges}
                                {hideNodeCard}
                                onTopKChange={handleTopKChange}
                                onComponentGapChange={handleComponentGapChange}
                                onLayerGapChange={handleLayerGapChange}
                                onNormalizeChange={handleNormalizeChange}
                                onCiThresholdChange={handleCiThresholdChange}
                                onHideUnpinnedEdgesChange={(v) => (hideUnpinnedEdges = v)}
                                onHideNodeCardChange={(v) => (hideNodeCard = v)}
                                {runningIntervention}
                                {generatingSubgraph}
                                onSelectionChange={handleComposerSelectionChange}
                                onRunIntervention={handleRunIntervention}
                                onSelectRun={handleSelectRun}
                                onDeleteRun={handleDeleteRun}
                                onForkRun={handleForkRun}
                                onDeleteFork={handleDeleteFork}
                                onGenerateGraphFromSelection={handleGenerateGraphFromSelection}
                            />
                        {/if}
                    {:else}
                        <!-- No graph yet -->
                        {#if graphCompute.status === "error"}
                            <div class="error-banner">
                                {graphCompute.error}
                                <button onclick={() => (graphCompute = { status: "idle" })}>Dismiss</button>
                                <button onclick={() => computeGraphForCard()}>Retry</button>
                            </div>
                        {/if}

                        <div
                            class="graph-area"
                            class:loading={graphCompute.status === "computing" && graphCompute.cardId === activeCard.id}
                        >
                            {#if graphCompute.status === "computing" && graphCompute.cardId === activeCard.id}
                                <ComputeProgressOverlay state={graphCompute.progress} />
                            {:else}
                                <div class="empty-state">
                                    <p>Click <strong>Compute</strong> to generate the attribution graph</p>
                                </div>
                            {/if}
                        </div>
                    {/if}
                {:else if promptCardLoading.status === "loading"}
                    <div class="empty-state">
                        <p>Loading prompt...</p>
                    </div>
                {:else if promptCardLoading.status === "error"}
                    <div class="empty-state">
                        <p class="error-text">Error loading prompt: {promptCardLoading.error}</p>
                        <button onclick={() => (promptCardLoading = { status: "idle" })}>Dismiss</button>
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
</div>

<style>
    .prompt-attributions-tab {
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

    .graph-info {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .graph-info strong {
        color: var(--text-secondary);
        font-weight: 500;
    }

    .graph-info .pinned-count {
        color: var(--accent-primary);
    }

    .graph-area {
        display: flex;
        flex-direction: column;
        position: relative;
        min-height: 400px;
        border: 1px solid var(--border-default);
        overflow: hidden;
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

    .empty-state .error-text {
        color: var(--status-negative-bright);
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
