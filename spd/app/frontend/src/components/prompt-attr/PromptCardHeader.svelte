<script lang="ts">
    import type { PromptCard } from "./types";

    type Props = {
        card: PromptCard;
        onSelectGraph: (graphId: number) => void;
        onCloseGraph: (graphId: number) => void;
        onNewGraph: () => void;
    };

    let { card, onSelectGraph, onCloseGraph, onNewGraph }: Props = $props();

    const activeGraph = $derived(card.graphs.find((g) => g.id === card.activeGraphId) ?? null);
    const isNewGraphMode = $derived(card.activeGraphId === null);

    // Config for displaying existing optimized graph params (read-only)
    const displayConfig = $derived.by(() => {
        if (!activeGraph?.data.optimization) return null;
        const opt = activeGraph.data.optimization;
        return {
            impMinCoeff: opt.imp_min_coeff,
            steps: opt.steps,
            pnorm: opt.pnorm,
            beta: opt.beta,
            ceLossCoeff: opt.ce_loss_coeff ?? 0,
            klLossCoeff: opt.kl_loss_coeff ?? 0,
            lossSeqPos: opt.loss_seq_pos,
            labelTokenId: opt.label_token,
            labelTokenText: opt.label_str ?? "",
            maskType: opt.mask_type ?? "stochastic",
        };
    });
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

    {#if displayConfig}
        <div class="opt-params-display">
            <span class="param"><span class="key">steps</span><span class="val">{displayConfig.steps}</span></span>
            <span class="param"
                ><span class="key">imp_min</span><span class="val">{displayConfig.impMinCoeff}</span></span
            >
            <span class="param"><span class="key">pnorm</span><span class="val">{displayConfig.pnorm}</span></span>
            <span class="param"><span class="key">beta</span><span class="val">{displayConfig.beta}</span></span>
            {#if displayConfig.ceLossCoeff > 0}
                <span class="param"
                    ><span class="key">ce</span><span class="val">{displayConfig.ceLossCoeff}</span></span
                >
                {#if displayConfig.labelTokenId !== null}
                    <span class="param"
                        ><span class="key">label</span><span class="val token">{displayConfig.labelTokenText}</span
                        ></span
                    >
                {/if}
            {/if}
            {#if displayConfig.klLossCoeff > 0}
                <span class="param"
                    ><span class="key">kl</span><span class="val">{displayConfig.klLossCoeff}</span></span
                >
            {/if}
            {#if displayConfig.ceLossCoeff > 0 || displayConfig.klLossCoeff > 0}
                <span class="param"
                    ><span class="key">seq_pos</span><span class="val"
                        >{displayConfig.lossSeqPos}{#if displayConfig.lossSeqPos >= 0 && displayConfig.lossSeqPos < card.tokens.length}<span
                                class="token">{card.tokens[displayConfig.lossSeqPos]}</span
                            >{/if}</span
                    ></span
                >
            {/if}
            <span class="param"><span class="key">mask</span><span class="val">{displayConfig.maskType}</span></span>
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

    .divider-line {
        border-top: 1px solid var(--border-default);
    }

    .opt-params-display {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
    }

    .opt-params-display .param {
        display: flex;
        align-items: center;
        gap: 2px;
    }

    .opt-params-display .key {
        color: var(--text-muted);
    }

    .opt-params-display .key::after {
        content: ":";
    }

    .opt-params-display .val {
        color: var(--text-secondary);
    }

    .opt-params-display .token {
        padding: 0 3px;
        background: var(--bg-inset);
        border: 1px solid var(--border-subtle);
        white-space: pre;
        margin-left: 2px;
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
</style>
