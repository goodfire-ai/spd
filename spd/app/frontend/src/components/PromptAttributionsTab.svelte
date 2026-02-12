<script lang="ts">
    import { getContext } from "svelte";
    import * as api from "../lib/api";
    import ProbColoredTokens from "./ProbColoredTokens.svelte";
    import {
        extractComponentKeys,
        filterInterventableNodes,
        type GraphData,
        type PinnedNode,
        type PromptPreview,
        type TokenInfo,
    } from "../lib/promptAttributionsTypes";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ComputeProgressOverlay from "./prompt-attr/ComputeProgressOverlay.svelte";
    import GraphTabs from "./prompt-attr/GraphTabs.svelte";
    import InterventionsView from "./prompt-attr/InterventionsView.svelte";
    import OptimizationParams from "./prompt-attr/OptimizationParams.svelte";
    import OptimizationSettings from "./prompt-attr/OptimizationSettings.svelte";
    import PromptTabs from "./prompt-attr/PromptTabs.svelte";
    import StagedNodesPanel from "./prompt-attr/StagedNodesPanel.svelte";
    import {
        defaultDraftState,
        defaultOptimizeConfig,
        isOptimizeConfigValid,
        validateOptimizeConfig,
        type ComposerState,
        type DraftState,
        type GraphComputeState,
        type OptimizeConfigDraft,
        type PromptCard,
        type StoredGraph,
        type TabViewState,
        type ViewSettings,
    } from "./prompt-attr/types";
    import ViewControls from "./prompt-attr/ViewControls.svelte";
    import ViewTabs from "./prompt-attr/ViewTabs.svelte";
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
                return `Manual (${Object.keys(data.nodeCiVals).length} components)`;
        }
    }

    type Props = {
        prompts: PromptPreview[];
        allTokens: TokenInfo[];
    };

    let { prompts, allTokens }: Props = $props();

    // Prompt cards state
    let promptCards = $state<PromptCard[]>([]);

    // Tab view state - discriminated union makes invalid states unrepresentable
    let tabView = $state<TabViewState>({ view: "draft", draft: defaultDraftState() });

    // Timer for debounced tokenization (not part of view state since it's internal)
    let tokenizeDebounceTimer: ReturnType<typeof setTimeout> | null = null;

    // Graph computation state
    let graphCompute = $state<GraphComputeState>({ status: "idle" });

    // Intervention loading state
    let runningIntervention = $state(false);
    let generatingSubgraph = $state(false);

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
        topK: 1000,
        componentGap: 8,
        layerGap: 40,
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

    // Derived state from tabView
    const activeCardId = $derived(tabView.view === "card" ? tabView.cardId : null);
    const activeCard = $derived(
        activeCardId !== null ? (promptCards.find((c) => c.id === activeCardId) ?? null) : null,
    );
    const activeGraph = $derived.by(() => {
        if (!activeCard) return null;
        return activeCard.graphs.find((g) => g.id === activeCard.activeGraphId) ?? null;
    });

    // Check if a standard graph already exists for the active card
    const hasStandardGraph = $derived(activeCard?.graphs.some((g) => g.data.graphType === "standard") ?? false);

    // Check if compute button should be enabled (config must be valid for optimized graphs)
    const canCompute = $derived.by(() => {
        if (!activeCard) return false;
        const needsOptimized = hasStandardGraph || activeCard.useOptimized;
        if (!needsOptimized) return true; // Standard graph mode - always valid
        return isOptimizeConfigValid(activeCard.newGraphConfig);
    });

    // Helper to update draft state (only valid when in draft view)
    function updateDraft(partial: Partial<DraftState>) {
        if (tabView.view !== "draft") return;
        tabView = { view: "draft", draft: { ...tabView.draft, ...partial } };
    }

    async function addPromptCard(
        promptId: number,
        tokens: string[],
        tokenIds: number[],
        nextTokenProbs: (number | null)[],
        isCustom: boolean,
    ) {
        tabView = { view: "loading" };
        try {
            await addPromptCardInner(promptId, tokens, tokenIds, nextTokenProbs, isCustom);
            tabView = { view: "card", cardId: promptId };
        } catch (error) {
            tabView = { view: "error", error: String(error) };
        }
    }

    async function addPromptCardInner(
        promptId: number,
        tokens: string[],
        tokenIds: number[],
        nextTokenProbs: (number | null)[],
        isCustom: boolean,
    ) {
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

        // Prefetch component data for all components in loaded graphs
        const allComponentKeys = graphs.flatMap((g) => extractComponentKeys(g.data));
        const uniqueKeys = [...new Set(allComponentKeys)];
        if (uniqueKeys.length > 0) {
            await runState.prefetchComponentData(uniqueKeys);
        }

        const newCard: PromptCard = {
            id: promptId,
            tokens,
            tokenIds,
            nextTokenProbs,
            isCustom,
            graphs,
            activeGraphId: graphs.length > 0 ? graphs[0].id : null,
            activeView: "graph",
            newGraphConfig: defaultOptimizeConfig(tokens.length),
            useOptimized: false,
        };
        promptCards = [...promptCards, newCard];
    }

    function handleSelectPrompt(prompt: PromptPreview) {
        // If prompt is already open as a card, just focus it
        const existingCard = promptCards.find((c) => c.id === prompt.id);
        if (existingCard) {
            tabView = { view: "card", cardId: prompt.id };
            return;
        }
        addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, prompt.next_token_probs, false);
    }

    // Create a new prompt from draft text and add as card
    async function handleAddFromDraft() {
        if (tabView.view !== "draft") return;
        const draftText = tabView.draft.text;
        if (!draftText.trim()) return;

        updateDraft({ isAdding: true });
        try {
            const prompt = await api.createCustomPrompt(draftText);
            // If prompt already exists (returned existing ID), just focus it
            const existingCard = promptCards.find((c) => c.id === prompt.id);
            if (existingCard) {
                tabView = { view: "card", cardId: prompt.id };
                return;
            }
            await addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, prompt.next_token_probs, true);
            // addPromptCard sets tabView to card on success
        } catch (error) {
            // If addPromptCard failed, we're already in error state
            // If createCustomPrompt failed, stay in draft and show error
            if (tabView.view === "draft") {
                updateDraft({ isAdding: false });
            }
            throw error;
        }
    }

    function handleStartNewDraft() {
        tabView = { view: "draft", draft: defaultDraftState() };
    }

    function handleDraftTextChange(text: string) {
        if (tabView.view !== "draft") return;
        updateDraft({ text });

        // Debounced tokenization for preview
        if (tokenizeDebounceTimer) clearTimeout(tokenizeDebounceTimer);
        if (!text.trim()) {
            updateDraft({ tokenPreview: { status: "uninitialized" } });
            return;
        }
        updateDraft({ tokenPreview: { status: "loading" } });
        tokenizeDebounceTimer = setTimeout(async () => {
            try {
                const result = await api.tokenizeText(text);
                updateDraft({
                    tokenPreview: {
                        status: "loaded",
                        data: { tokens: result.tokens, next_token_probs: result.next_token_probs },
                    },
                });
            } catch (e) {
                updateDraft({
                    tokenPreview: {
                        status: "error",
                        error: e instanceof Error ? e.message : String(e),
                    },
                });
            }
        }, 150);
    }

    function handleDraftKeydown(e: KeyboardEvent) {
        // Enter without shift = add prompt, Shift+Enter = newline
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleAddFromDraft();
        }
    }

    function handleCloseCard(cardId: number) {
        promptCards = promptCards.filter((c) => c.id !== cardId);
        if (activeCardId === cardId) {
            // Switch to another card or back to draft
            if (promptCards.length > 0) {
                tabView = { view: "card", cardId: promptCards[promptCards.length - 1].id };
            } else {
                tabView = { view: "draft", draft: defaultDraftState() };
            }
        }
    }

    function handleSelectCard(cardId: number) {
        tabView = { view: "card", cardId };
    }

    function handleDismissError() {
        // Go back to draft on error dismissal
        tabView = { view: "draft", draft: defaultDraftState() };
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

    function handleOptimizeConfigChange(newConfig: OptimizeConfigDraft) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) =>
            card.id === activeCard.id ? { ...card, newGraphConfig: newConfig } : card,
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
                stages: [{ name: "Computing attribution graph from selection", progress: 0 }],
                currentStage: 0,
            },
        };

        try {
            const data = await api.computeGraphStream(
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

            // Prefetch component data for all components in the subgraph
            const componentKeys = extractComponentKeys(data);
            await runState.prefetchComponentData(componentKeys);

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

        const draftConfig = activeCard.newGraphConfig;
        // If a standard graph exists, always use optimized (no point computing another standard)
        const isOptimized = hasStandardGraph || activeCard.useOptimized;
        const cardId = activeCard.id;

        // Validate config (button should be disabled if invalid, so this is a safety check)
        const validConfig = validateOptimizeConfig(draftConfig);
        if (isOptimized && !validConfig) {
            throw new Error("Invalid config: CE loss requires a target token");
        }

        const initialProgress = isOptimized
            ? {
                  stages: [
                      { name: "Optimizing", progress: 0 },
                      { name: "Computing attribution graph", progress: 0 },
                  ],
                  currentStage: 0,
              }
            : {
                  stages: [{ name: "Computing attribution graph", progress: 0 }],
                  currentStage: 0,
              };

        graphCompute = { status: "computing", cardId, progress: initialProgress };

        try {
            let data: GraphData;

            if (isOptimized) {
                // validConfig is guaranteed non-null here due to early return above
                const optConfig = validConfig!;

                const params: api.ComputeGraphOptimizedParams = {
                    promptId: cardId,
                    normalize: defaultViewSettings.normalizeEdges,
                    impMinCoeff: optConfig.impMinCoeff,
                    steps: optConfig.steps,
                    pnorm: optConfig.pnorm,
                    beta: optConfig.beta,
                    ciThreshold: defaultViewSettings.ciThreshold,
                    maskType: optConfig.maskType,
                    lossType: optConfig.loss.type,
                    lossCoeff: optConfig.loss.coeff,
                    lossPosition: optConfig.loss.position,
                    labelToken: optConfig.loss.type === "ce" ? optConfig.loss.labelTokenId : undefined,
                    advPgdNSteps:
                        optConfig.advPgdNSteps !== null && optConfig.advPgdStepSize !== null
                            ? optConfig.advPgdNSteps
                            : undefined,
                    advPgdStepSize:
                        optConfig.advPgdNSteps !== null && optConfig.advPgdStepSize !== null
                            ? optConfig.advPgdStepSize
                            : undefined,
                };

                data = await api.computeGraphOptimizedStream(params, (progress) => {
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
                data = await api.computeGraphStream(params, (progress) => {
                    if (graphCompute.status !== "computing") return;
                    graphCompute.progress.stages[0].progress = progress.current / progress.total;
                });
            }

            // Initialize composer state for the new graph
            getComposerState(data.id, Object.keys(data.nodeCiVals));

            // Prefetch component data BEFORE state update so cache is populated when components mount
            const componentKeys = extractComponentKeys(data);
            await runState.prefetchComponentData(componentKeys);

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
</script>

<div class="prompt-attributions-tab">
    <div class="main-content">
        <div class="graph-container">
            <div class="card-tabs-row">
                <PromptTabs
                    cards={promptCards}
                    {tabView}
                    onSelectCard={handleSelectCard}
                    onCloseCard={handleCloseCard}
                    onSelectDraft={handleStartNewDraft}
                    onAddClick={handleStartNewDraft}
                />
            </div>

            <div class="card-content">
                {#if tabView.view === "draft"}
                    {@const draft = tabView.draft}
                    <!-- New prompt staging area -->
                    <div class="draft-staging">
                        <div class="draft-main">
                            <div class="draft-input-section">
                                <label class="draft-label">Enter prompt text</label>
                                <textarea
                                    class="draft-textarea"
                                    placeholder="Type your prompt here... (Enter to add)"
                                    value={draft.text}
                                    oninput={(e) => handleDraftTextChange(e.currentTarget.value)}
                                    onkeydown={handleDraftKeydown}
                                    rows={2}
                                ></textarea>
                                {#if draft.tokenPreview.status === "loading"}
                                    <div class="token-preview-row loading">Tokenizing...</div>
                                {:else if draft.tokenPreview.status === "error"}
                                    <div class="token-preview-row error">{draft.tokenPreview.error}</div>
                                {:else if draft.tokenPreview.status === "loaded" && draft.tokenPreview.data.tokens.length > 0}
                                    {@const { tokens, next_token_probs } = draft.tokenPreview.data}
                                    <div class="token-preview-row">
                                        <ProbColoredTokens {tokens} nextTokenProbs={next_token_probs} />
                                        <span class="token-count">{tokens.length} tokens</span>
                                    </div>
                                {/if}
                                <button
                                    class="btn-add-prompt"
                                    onclick={handleAddFromDraft}
                                    disabled={!draft.text.trim() || draft.isAdding}
                                >
                                    {draft.isAdding ? "Adding..." : "Add Prompt"}
                                </button>
                            </div>

                            {#if prompts.length > 0}
                                <div class="existing-prompts-section">
                                    <label class="draft-label">Or select existing ({prompts.length})</label>
                                    <div class="prompt-list">
                                        {#each prompts as prompt (prompt.id)}
                                            <button class="prompt-item" onclick={() => handleSelectPrompt(prompt)}>
                                                <span class="prompt-id">#{prompt.id}</span>
                                                <span class="prompt-text">{prompt.preview}</span>
                                            </button>
                                        {/each}
                                    </div>
                                </div>
                            {/if}
                        </div>
                    </div>
                {:else if activeCard}
                    <!-- Level 1: Tokens -->
                    <div class="prompt-tokens">
                        <ProbColoredTokens tokens={activeCard.tokens} nextTokenProbs={activeCard.nextTokenProbs} />
                    </div>

                    <!-- Level 2: Graph tabs -->
                    <GraphTabs
                        graphs={activeCard.graphs}
                        activeGraphId={activeCard.activeGraphId}
                        onSelectGraph={handleSelectGraph}
                        onCloseGraph={handleCloseGraph}
                        onNewGraph={handleEnterNewGraphMode}
                    />

                    {#if activeGraph}
                        <!-- Optimization params (if optimized graph) -->
                        {#if activeGraph.data.optimization}
                            <OptimizationParams
                                optimization={activeGraph.data.optimization}
                                tokens={activeCard.tokens}
                            />
                        {/if}

                        <!-- Level 3: View tabs -->
                        <div>
                            <ViewTabs
                                activeView={activeCard.activeView}
                                interventionRunCount={activeGraph.interventionRuns.length}
                                onViewChange={handleViewChange}
                            />

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
                                            tokenIds={activeCard.tokenIds}
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
                                    nodeCiVals={activeGraph.data.nodeCiVals}
                                    nodeSubcompActs={activeGraph.data.nodeSubcompActs}
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
                        </div>
                    {:else}
                        <!-- No graph yet -->
                        {#if graphCompute.status === "error"}
                            <div class="error-banner">
                                {graphCompute.error}
                                <button onclick={() => (graphCompute = { status: "idle" })}>Dismiss</button>
                                <button onclick={() => computeGraphForCard()} disabled={!canCompute}>Retry</button>
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
                                    <div class="compute-controls">
                                        {#if !hasStandardGraph}
                                            <label class="optimize-checkbox">
                                                <input
                                                    type="checkbox"
                                                    checked={activeCard.useOptimized}
                                                    onchange={(e) => handleUseOptimizedChange(e.currentTarget.checked)}
                                                />
                                                <span>Optimize</span>
                                            </label>
                                        {/if}
                                        {#if hasStandardGraph || activeCard.useOptimized}
                                            <OptimizationSettings
                                                config={activeCard.newGraphConfig}
                                                tokens={activeCard.tokens}
                                                {allTokens}
                                                onChange={handleOptimizeConfigChange}
                                                cardId={activeCard.id}
                                            />
                                        {/if}
                                        <button
                                            class="btn-compute-center"
                                            onclick={() => computeGraphForCard()}
                                            disabled={!canCompute}
                                        >
                                            Compute
                                        </button>
                                    </div>
                                </div>
                            {/if}
                        </div>
                    {/if}
                {:else if tabView.view === "loading"}
                    <div class="empty-state">
                        <p>Loading prompt...</p>
                    </div>
                {:else if tabView.view === "error"}
                    <div class="empty-state">
                        <p class="error-text">Error loading prompt: {tabView.error}</p>
                        <button onclick={handleDismissError}>Dismiss</button>
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

    .prompt-tokens {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        align-items: center;
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

    .empty-state .error-text {
        color: var(--status-negative-bright);
    }

    .btn-compute-center {
        padding: var(--space-2) var(--space-4);
        background: var(--bg-elevated);
        border: 1px dashed var(--accent-primary-dim);
        font-size: var(--text-base);
        font-family: var(--font-mono);
        font-weight: 500;
        color: var(--accent-primary);
        cursor: pointer;
    }

    .btn-compute-center:hover {
        background: var(--bg-inset);
        border-style: solid;
        border-color: var(--accent-primary);
    }

    .compute-controls {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--space-3);
    }

    .optimize-checkbox {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        cursor: pointer;
    }

    .optimize-checkbox:hover {
        color: var(--text-primary);
    }

    .optimize-checkbox input {
        cursor: pointer;
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

    /* Draft staging area styles */
    .draft-staging {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: var(--space-6);
    }

    .draft-main {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-6);
        max-width: 900px;
        width: 100%;
        align-items: start;
    }

    .draft-input-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .draft-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
    }

    .draft-textarea {
        width: 100%;
        padding: var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        resize: vertical;
        min-height: 120px;
    }

    .draft-textarea:focus {
        outline: none;
        border-color: var(--accent-primary);
    }

    .draft-textarea::placeholder {
        color: var(--text-muted);
    }

    .btn-add-prompt {
        align-self: flex-start;
        padding: var(--space-1) var(--space-3);
        background: var(--accent-primary);
        border: none;
        color: white;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 500;
        cursor: pointer;
    }

    .btn-add-prompt:hover:not(:disabled) {
        background: var(--accent-primary-bright);
    }

    .btn-add-prompt:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .token-preview-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        align-items: center;
    }

    .token-preview-row.loading {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .token-preview-row.error {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--status-negative);
    }

    .token-preview-row .token-count {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
        margin-left: var(--space-2);
    }

    .existing-prompts-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .prompt-list {
        display: flex;
        flex-direction: column;
        max-height: 400px;
        overflow-y: auto;
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
    }

    .prompt-item {
        width: 100%;
        padding: var(--space-2) var(--space-3);
        background: transparent;
        border: none;
        border-bottom: 1px solid var(--border-subtle);
        cursor: pointer;
        text-align: left;
        display: flex;
        gap: var(--space-2);
        align-items: baseline;
        color: var(--text-primary);
    }

    .prompt-item:last-child {
        border-bottom: none;
    }

    .prompt-item:hover {
        background: var(--bg-surface);
    }

    .prompt-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .prompt-text {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--text-primary);
    }
</style>
