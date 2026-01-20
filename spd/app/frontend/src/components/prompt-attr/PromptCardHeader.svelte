<script lang="ts">
    import type { TokenInfo } from "../../lib/promptAttributionsTypes";
    import type { PromptCard, OptimizeConfig, StoredGraph, MaskType } from "./types";
    import TokenDropdown from "./TokenDropdown.svelte";

    type Props = {
        card: PromptCard;
        isLoading: boolean;
        tokens: TokenInfo[];
        onUseOptimizedChange: (useOptimized: boolean) => void;
        onOptimizeConfigChange: (update: Partial<OptimizeConfig>) => void;
        onCompute: () => void;
        onSelectGraph: (graphId: number) => void;
        onCloseGraph: (graphId: number) => void;
        onNewGraph: () => void;
    };

    let {
        card,
        isLoading,
        tokens,
        onUseOptimizedChange,
        onOptimizeConfigChange,
        onCompute,
        onSelectGraph,
        onCloseGraph,
        onNewGraph,
    }: Props = $props();

    // Determine mode: viewing existing graph vs creating new
    const activeGraph = $derived(card.graphs.find((g) => g.id === card.activeGraphId) ?? null);
    const isNewGraphMode = $derived(card.activeGraphId === null);

    // Config depends on mode - either from existing graph or from newGraphConfig
    const displayConfig = $derived.by(() => {
        if (activeGraph?.data.optimization) {
            const opt = activeGraph.data.optimization;
            return {
                impMinCoeff: opt.imp_min_coeff,
                steps: opt.steps,
                pnorm: opt.pnorm,
                beta: opt.beta,
                ceLossCoeff: opt.ce_loss_coeff ?? 0,
                klLossCoeff: opt.kl_loss_coeff ?? 0,
                labelTokenId: opt.label_token,
                labelTokenText: opt.label_str ?? "",
                labelTokenPreview: opt.label_str,
                maskType: opt.mask_type ?? "stochastic",
            };
        }
        return card.newGraphConfig;
    });

    // For new graph mode, use card config
    const optConfig = $derived(card.newGraphConfig);
    const useOptimized = $derived(card.useOptimized);

    // Derived: whether each loss type is active (for validation)
    const useCE = $derived(optConfig.ceLossCoeff > 0 && optConfig.labelTokenId !== null);
    const useKL = $derived(optConfig.klLossCoeff > 0);

    // Validation for optimized mode
    const canCompute = $derived.by(() => {
        if (!isNewGraphMode) return false; // Can't compute when viewing existing graph
        if (isLoading) return false;
        if (!useOptimized) return true;

        // Must have at least one active loss
        if (!useCE && !useKL) return false;

        // ce_coeff > 0 requires label token
        if (optConfig.ceLossCoeff > 0 && optConfig.labelTokenId === null) return false;

        // label token set requires ce_coeff > 0
        if (optConfig.labelTokenId !== null && optConfig.ceLossCoeff <= 0) return false;

        return true;
    });

    const buttonText = $derived.by(() => {
        if (isLoading) return "Computing...";
        if (!useOptimized) return "Compute";
        // Optimized mode validation messages
        if (!useCE && !useKL) return "Set a loss coeff > 0";
        if (optConfig.ceLossCoeff > 0 && optConfig.labelTokenId === null) return "Enter label token";
        if (optConfig.labelTokenId !== null && optConfig.ceLossCoeff <= 0) return "Set ce_coeff > 0";
        return "Compute";
    });

    // Check if we already have a matching standard graph
    const hasStandardGraph = $derived(card.graphs.some((g) => !g.data.optimization));

    // Check if we already have a matching optimized graph with same params
    function hasMatchingOptimizedGraph(graphs: StoredGraph[], config: OptimizeConfig): boolean {
        return graphs.some((g) => {
            const opt = g.data.optimization;
            if (!opt) return false;
            // Compare all relevant params
            const stepsMatch = opt.steps === config.steps;
            const impMinMatch = Math.abs(opt.imp_min_coeff - config.impMinCoeff) < 0.0000001;
            const pnormMatch = Math.abs(opt.pnorm - config.pnorm) < 0.0000001;
            const betaMatch = Math.abs(opt.beta - config.beta) < 0.0000001;
            const ceMatch = (opt.ce_loss_coeff ?? 0) === config.ceLossCoeff && opt.label_token === config.labelTokenId;
            const klMatch = (opt.kl_loss_coeff ?? 0) === config.klLossCoeff;
            const maskTypeMatch = (opt.mask_type ?? "stochastic") === config.maskType;
            return stepsMatch && impMinMatch && pnormMatch && betaMatch && ceMatch && klMatch && maskTypeMatch;
        });
    }

    const hasMatchingGraph = $derived.by(() => {
        if (!useOptimized) return hasStandardGraph;
        return hasMatchingOptimizedGraph(card.graphs, optConfig);
    });

    // Show compute controls when in new graph mode
    const showComputeButton = $derived(isNewGraphMode && !hasMatchingGraph);
</script>

<div class="staged-header">
    <div class="staged-tokens">
        {#each card.tokens as tok, i (i)}
            <span class="staged-token" class:custom={card.isCustom}>{tok}</span>
        {/each}
    </div>

    <div class="graph-tabs">
        {#each card.graphs as graph (graph.id)}
            <div class="graph-tab" class:active={graph.id === card.activeGraphId}>
                <button class="tab-label" onclick={() => onSelectGraph(graph.id)}>
                    {graph.label} <span class="graph-id">#{graph.id}</span>
                </button>
                <button class="tab-close" onclick={() => onCloseGraph(graph.id)}>×</button>
            </div>
        {/each}
        {#if card.graphs.length > 0}
            <button class="btn-new-graph" class:active={isNewGraphMode} onclick={onNewGraph} disabled={isNewGraphMode}>
                + New
            </button>
        {/if}
    </div>

    <div class="divider-line"></div>

    {#if isNewGraphMode}
        <!-- Editable config for creating new graph -->
        <div class="staged-controls">
            <div class="compute-options">
                <span class="info-icon" data-tooltip="Default output_prob_threshold=0.01">?</span>
                <label class="checkbox">
                    <input
                        type="checkbox"
                        checked={useOptimized}
                        onchange={(e) => onUseOptimizedChange(e.currentTarget.checked)}
                    />
                    <span>Optimize</span>
                </label>
                {#if useOptimized}
                    <!-- Common settings -->
                    <label>
                        <span>imp_min_coeff</span>
                        <input
                            type="number"
                            value={optConfig.impMinCoeff}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onOptimizeConfigChange({ impMinCoeff: parseFloat(e.currentTarget.value) });
                            }}
                            min={0.001}
                            max={10}
                            step={0.01}
                        />
                    </label>
                    <label>
                        <span>steps</span>
                        <input
                            type="number"
                            value={optConfig.steps}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onOptimizeConfigChange({ steps: parseInt(e.currentTarget.value) });
                            }}
                            min={10}
                            max={5000}
                            step={100}
                        />
                    </label>
                    <label>
                        <span>pnorm</span>
                        <input
                            type="number"
                            value={optConfig.pnorm}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onOptimizeConfigChange({ pnorm: parseFloat(e.currentTarget.value) });
                            }}
                            min={0.1}
                            max={2}
                            step={0.1}
                        />
                    </label>
                    <label>
                        <span>beta</span>
                        <input
                            type="number"
                            value={optConfig.beta}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onOptimizeConfigChange({ beta: parseFloat(e.currentTarget.value) });
                            }}
                            min={0}
                            max={10}
                            step={0.1}
                        />
                    </label>
                    <!-- CE Loss settings -->
                    <label>
                        <span>ce_coeff</span>
                        <input
                            type="number"
                            value={optConfig.ceLossCoeff}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onOptimizeConfigChange({ ceLossCoeff: parseFloat(e.currentTarget.value) });
                            }}
                            min={0}
                            step={0.1}
                        />
                    </label>
                    <label class="label-token-input">
                        <span>Label</span>
                        <TokenDropdown
                            {tokens}
                            value={optConfig.labelTokenText}
                            selectedTokenId={optConfig.labelTokenId}
                            onSelect={(tokenId, tokenString) => {
                                onOptimizeConfigChange({
                                    labelTokenText: tokenString,
                                    labelTokenId: tokenId,
                                    labelTokenPreview: tokenId !== null ? tokenString : "",
                                });
                            }}
                            placeholder="Search token..."
                        />
                        {#if optConfig.labelTokenId !== null}
                            <span class="token-id-hint">#{optConfig.labelTokenId}</span>
                        {/if}
                    </label>
                    <!-- KL Loss settings -->
                    <label>
                        <span>kl_coeff</span>
                        <input
                            type="number"
                            value={optConfig.klLossCoeff}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onOptimizeConfigChange({ klLossCoeff: parseFloat(e.currentTarget.value) });
                            }}
                            min={0}
                            step={0.1}
                        />
                    </label>
                    <!-- Mask type dropdown -->
                    <label>
                        <span>mask_type</span>
                        <select
                            value={optConfig.maskType}
                            onchange={(e) => onOptimizeConfigChange({ maskType: e.currentTarget.value as MaskType })}
                        >
                            <option value="stochastic">stochastic</option>
                            <option value="ci">ci</option>
                        </select>
                    </label>
                {/if}

                {#if showComputeButton}
                    <button class="btn-compute" onclick={onCompute} disabled={!canCompute}>
                        {buttonText}
                    </button>
                {:else if hasMatchingGraph}
                    <span class="matching-graph-hint">Graph already exists</span>
                {/if}
            </div>
        </div>
    {:else if activeGraph?.data.optimization}
        <!-- Read-only display of existing optimized graph params -->
        <div class="staged-controls readonly">
            <div class="compute-options">
                <span class="readonly-label">Optimized graph params:</span>
                <label>
                    <span>imp_min_coeff</span>
                    <input type="number" value={displayConfig.impMinCoeff} disabled />
                </label>
                <label>
                    <span>steps</span>
                    <input type="number" value={displayConfig.steps} disabled />
                </label>
                <label>
                    <span>pnorm</span>
                    <input type="number" value={displayConfig.pnorm} disabled />
                </label>
                <label>
                    <span>beta</span>
                    <input type="number" value={displayConfig.beta} disabled />
                </label>
                {#if displayConfig.ceLossCoeff > 0}
                    <label>
                        <span>ce_coeff</span>
                        <input type="number" value={displayConfig.ceLossCoeff} disabled />
                    </label>
                    {#if displayConfig.labelTokenId !== null}
                        <label>
                            <span>Label</span>
                            <input type="text" value={displayConfig.labelTokenText} disabled />
                            <span class="token-id-hint">#{displayConfig.labelTokenId}</span>
                        </label>
                    {/if}
                {/if}
                {#if displayConfig.klLossCoeff > 0}
                    <label>
                        <span>kl_coeff</span>
                        <input type="number" value={displayConfig.klLossCoeff} disabled />
                    </label>
                {/if}
                <label>
                    <span>mask_type</span>
                    <input type="text" value={displayConfig.maskType} disabled />
                </label>
            </div>
        </div>
    {:else if activeGraph}
        <!-- Standard graph - no optimization params to show -->
        <div class="staged-controls readonly">
            <span class="readonly-label">Standard graph (no optimization)</span>
        </div>
    {/if}
</div>

<style>
    .staged-header {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .staged-tokens {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        align-items: center;
    }

    .staged-token {
        padding: 2px 3px;
        background: var(--bg-elevated);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-primary);
        white-space: pre;
        border: 1px solid var(--border-subtle);
    }

    .staged-token.custom {
        background: var(--bg-inset);
        border-color: var(--status-info);
        color: var(--status-info-bright);
    }

    .staged-controls {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-4);
    }

    .divider-line {
        border-top: 1px solid var(--border-default);
    }

    .compute-options {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        flex-wrap: wrap;
    }

    .compute-options label {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
    }

    .compute-options label span {
        font-weight: 500;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    .compute-options input[type="number"] {
        width: 110px;
        padding: var(--space-1);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .compute-options input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .compute-options select {
        padding: var(--space-1);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        cursor: pointer;
    }

    .compute-options select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .compute-options label.checkbox {
        gap: var(--space-1);
    }

    .compute-options .label-token-input {
        flex-wrap: wrap;
    }

    .token-id-hint {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .btn-compute {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px dashed var(--accent-primary-dim);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        font-weight: 500;
        white-space: nowrap;
        color: var(--accent-primary);
        cursor: pointer;
    }

    .btn-compute:hover:not(:disabled) {
        background: var(--bg-inset);
        border-style: solid;
        border-color: var(--accent-primary);
    }

    .btn-compute:disabled {
        background: var(--bg-elevated);
        border-color: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .graph-tabs {
        display: flex;
        gap: var(--space-1);
    }

    .graph-tab {
        display: flex;
        align-items: center;
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .graph-tab:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .graph-tab.active {
        background: var(--accent-primary);
        color: var(--bg-base);
    }

    .tab-label {
        padding: var(--space-1) var(--space-2);
        background: transparent;
        border: none;
        font-size: inherit;
        font-family: inherit;
        color: inherit;
        cursor: pointer;
    }

    .graph-id {
        font-size: var(--text-xs);
        color: var(--text-muted);
        opacity: 0.7;
    }

    .tab-close {
        padding: var(--space-1);
        background: transparent;
        border: none;
        border-left: 1px solid var(--border-subtle);
        font-size: var(--text-sm);
        line-height: 1;
        opacity: 0.6;
        cursor: pointer;
        color: inherit;
    }

    .graph-tab.active .tab-close {
        border-left-color: var(--accent-primary);
    }

    .tab-close:hover {
        opacity: 1;
        color: var(--status-negative-bright);
    }

    .btn-new-graph {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px dashed var(--border-default);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        font-weight: 500;
        cursor: pointer;
    }

    .btn-new-graph:hover:not(:disabled) {
        background: var(--bg-inset);
        border-style: solid;
        border-color: var(--accent-primary-dim);
        color: var(--accent-primary);
    }

    .btn-new-graph.active {
        background: var(--accent-primary);
        color: var(--bg-base);
        border-style: solid;
        border-color: var(--accent-primary);
    }

    .btn-new-graph:disabled {
        cursor: default;
    }

    .staged-controls.readonly {
        background: var(--bg-inset);
        /* padding: var(--space-2); */
    }

    .staged-controls.readonly input {
        background: var(--bg-elevated);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .readonly-label {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        margin-right: var(--space-2);
    }

    .matching-graph-hint {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
    }

    .compute-options input[type="text"] {
        width: 110px;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .staged-controls.readonly .compute-options input[type="text"] {
        color: var(--text-muted);
        cursor: not-allowed;
    }
</style>
