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
                    oninput={(e) => {
                        if (e.currentTarget.value === "") return;
                        onOptionsChange({ maxMeanCI: parseFloat(e.currentTarget.value) });
                    }}
                    min={0}
                    max={1}
                    step={0.01}
                />
            </label>
            <span class="info-icon" data-tooltip="Backend defaults: ci_threshold=1e-6, output_prob_threshold=0.01"
                >?</span
            >
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
                    <span>ce</span>
                    <input
                        type="number"
                        value={optConfig.ceLossCoeff}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onOptimizeConfigChange({ ceLossCoeff: parseFloat(e.currentTarget.value) });
                        }}
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
                        max={1}
                        step={0.1}
                    />
                </label>
            {/if}
        </div>

    </div>

    <div class="graph-tabs">
        {#each card.graphs as graph (graph.id)}
            <div class="graph-tab" class:active={graph.id === card.activeGraphId}>
                <button class="tab-label" onclick={() => onSelectGraph(graph.id)}>
                    {graph.label}
                </button>
                <button class="tab-close" onclick={() => onCloseGraph(graph.id)}>×</button>
            </div>
        {/each}
        <button class="btn-compute" onclick={onCompute} disabled={!canCompute}>
            {buttonText}
        </button>
    </div>
</div>

<style>
    .staged-header {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        /* background: var(--bg-surface); */
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

    .compute-options {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        flex-wrap: wrap;
    }

    .info-icon {
        position: relative;
        cursor: help;
        color: var(--text-muted);
        font-size: var(--text-xs);
        width: 14px;
        height: 14px;
        line-height: 14px;
        text-align: center;
        border: 1px solid var(--border-default);
        border-radius: 50%;
    }

    .info-icon::after {
        content: attr(data-tooltip);
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        margin-top: 4px;
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 100;
    }

    .info-icon:hover::after {
        opacity: 1;
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
        width: 55px;
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

    .compute-options label.checkbox {
        gap: var(--space-1);
    }

    .compute-options .label-token-input {
        flex-wrap: wrap;
    }

    .compute-options .label-token-input .text-input {
        width: 80px;
        padding: var(--space-1);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .compute-options .label-token-input .text-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .compute-options .token-preview {
        font-size: var(--text-xs);
        color: var(--status-positive-bright);
        font-family: var(--font-mono);
    }

    .compute-options .token-preview.error {
        color: var(--status-negative-bright);
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
</style>
