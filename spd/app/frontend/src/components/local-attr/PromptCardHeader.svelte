<script lang="ts">
    import type { PromptCard, ComputeOptions, OptimizeConfig } from "./types";

    type Props = {
        card: PromptCard;
        options: ComputeOptions;
        isLoading: boolean;
        onOptionsChange: (update: Partial<ComputeOptions>) => void;
        onOptimizeConfigChange: (update: Partial<OptimizeConfig>) => void;
        onCompute: () => void;
        onSelectGraph: (graphId: string) => void;
        onCloseGraph: (graphId: string) => void;
    };

    let {
        card,
        options,
        isLoading,
        onOptionsChange,
        onOptimizeConfigChange,
        onCompute,
        onSelectGraph,
        onCloseGraph,
    }: Props = $props();

    const optConfig = $derived(options.optimizeConfig);

    const canCompute = $derived(!isLoading && (!options.useOptimized || optConfig.labelTokenId !== null));

    const buttonText = $derived.by(() => {
        if (isLoading) {
            return "Computing...";
        }
        if (options.useOptimized && !optConfig.labelTokenId) {
            return "Enter label token";
        }
        return options.useOptimized ? "Compute (Optimized)" : "Compute";
    });
</script>

<div class="staged-header">
    <div class="staged-tokens">
        {#each card.tokens as tok, i (i)}
            <span class="staged-token" class:custom={card.isCustom}>{tok}</span>
        {/each}
    </div>

    <div class="staged-controls">
        <div class="compute-options">
            <label>
                <span>Max CI</span>
                <input
                    type="number"
                    value={options.maxMeanCI}
                    oninput={(e) => onOptionsChange({ maxMeanCI: parseFloat(e.currentTarget.value) || 1.0 })}
                    min={0}
                    max={1}
                    step={0.01}
                />
            </label>
            <label class="checkbox">
                <input
                    type="checkbox"
                    checked={options.normalizeEdges}
                    onchange={(e) => onOptionsChange({ normalizeEdges: e.currentTarget.checked })}
                />
                <span>Normalize</span>
            </label>
            <label class="checkbox">
                <input
                    type="checkbox"
                    checked={options.useOptimized}
                    onchange={(e) => onOptionsChange({ useOptimized: e.currentTarget.checked })}
                />
                <span>Optimize</span>
            </label>
            {#if options.useOptimized}
                <label class="label-token-input">
                    <span>Label</span>
                    <input
                        type="text"
                        value={optConfig.labelTokenText}
                        oninput={(e) => onOptimizeConfigChange({ labelTokenText: e.currentTarget.value })}
                        placeholder="e.g. ' world'"
                        class="text-input"
                    />
                    {#if optConfig.labelTokenPreview}
                        <span class="token-preview" class:error={!optConfig.labelTokenId}>
                            → {optConfig.labelTokenPreview}
                        </span>
                    {/if}
                </label>
                <label>
                    <span>imp_min</span>
                    <input
                        type="number"
                        value={optConfig.impMinCoeff}
                        oninput={(e) =>
                            onOptimizeConfigChange({ impMinCoeff: parseFloat(e.currentTarget.value) || 0.1 })}
                        min={0.001}
                        max={10}
                        step={0.01}
                    />
                </label>
                <label>
                    <span>ce</span>
                    <input
                        type="number"
                        value={optConfig.ceLossCoeff}
                        oninput={(e) =>
                            onOptimizeConfigChange({ ceLossCoeff: parseFloat(e.currentTarget.value) || 1.0 })}
                        min={0.001}
                        max={10}
                        step={0.1}
                    />
                </label>
                <label>
                    <span>steps</span>
                    <input
                        type="number"
                        value={optConfig.steps}
                        oninput={(e) => onOptimizeConfigChange({ steps: parseInt(e.currentTarget.value) || 2000 })}
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
                        oninput={(e) => onOptimizeConfigChange({ pnorm: parseFloat(e.currentTarget.value) || 0.3 })}
                        min={0.1}
                        max={1}
                        step={0.1}
                    />
                </label>
            {/if}
        </div>

        <button class="btn-compute" onclick={onCompute} disabled={!canCompute}>
            {buttonText}
        </button>
    </div>

    {#if card.graphs.length > 0}
        <div class="graph-tabs">
            {#each card.graphs as graph (graph.id)}
                <div class="graph-tab" class:active={graph.id === card.activeGraphId}>
                    <button class="tab-label" onclick={() => onSelectGraph(graph.id)}>
                        {graph.label}
                    </button>
                    <button class="tab-close" onclick={() => onCloseGraph(graph.id)}>×</button>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
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
        padding: 2px 2px;
        background: #e8e8e8;
        border-radius: 3px;
        font-family: "SF Mono", Monaco, monospace;
        font-size: 0.8rem;
        color: #424242;
        white-space: pre;
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

    .compute-options .label-token-input {
        flex-wrap: wrap;
    }

    .compute-options .label-token-input .text-input {
        width: 80px;
        padding: 0.2rem 0.35rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: "SF Mono", Monaco, monospace;
    }

    .compute-options .token-preview {
        font-size: 0.75rem;
        color: #4caf50;
        font-family: "SF Mono", Monaco, monospace;
    }

    .compute-options .token-preview.error {
        color: #f44336;
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
</style>
