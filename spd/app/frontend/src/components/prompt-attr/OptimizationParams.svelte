<script lang="ts">
    import type { OptimizationResult } from "../../lib/promptAttributionsTypes";

    type Props = {
        optimization: OptimizationResult;
        tokens: string[];
    };

    let { optimization, tokens }: Props = $props();

    const config = $derived({
        impMinCoeff: optimization.imp_min_coeff,
        steps: optimization.steps,
        pnorm: optimization.pnorm,
        beta: optimization.beta,
        ceLossCoeff: optimization.ce_loss_coeff ?? 0,
        klLossCoeff: optimization.kl_loss_coeff ?? 0,
        lossSeqPos: optimization.loss_seq_pos,
        labelTokenId: optimization.label_token,
        labelTokenText: optimization.label_str ?? "",
        maskType: optimization.mask_type ?? "stochastic",
    });
</script>

<div class="opt-params">
    <div class="params-group">
        <span class="param"><span class="key">steps</span><span class="val">{config.steps}</span></span>
        <span class="param"><span class="key">imp_min</span><span class="val">{config.impMinCoeff}</span></span>
        <span class="param"><span class="key">pnorm</span><span class="val">{config.pnorm}</span></span>
        <span class="param"><span class="key">beta</span><span class="val">{config.beta}</span></span>
        <span class="param"><span class="key">mask</span><span class="val">{config.maskType}</span></span>
    </div>
    {#if config.klLossCoeff > 0}
        <div class="params-group loss-group">
            <span class="group-label">KL</span>
            <span class="param"><span class="key">coeff</span><span class="val">{config.klLossCoeff}</span></span>
        </div>
    {/if}
    {#if config.ceLossCoeff > 0}
        <div class="params-group loss-group">
            <span class="group-label">CE</span>
            <span class="param"><span class="key">coeff</span><span class="val">{config.ceLossCoeff}</span></span>
            <span class="ce-target">
                <span class="target-text">at pos</span>
                <span class="val">{config.lossSeqPos}</span>
                {#if config.lossSeqPos >= 0 && config.lossSeqPos < tokens.length}
                    <span class="token">{tokens[config.lossSeqPos]}</span>
                {/if}
                <span class="target-text">predict</span>
                {#if config.labelTokenId !== null}
                    <span class="token label-token">{config.labelTokenText}</span>
                {:else}
                    <span class="val muted">—</span>
                {/if}
            </span>
        </div>
    {/if}
</div>

<style>
    .opt-params {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        align-items: center;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
    }

    .params-group {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--space-2);
    }

    .loss-group {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
    }

    .group-label {
        font-weight: 600;
        color: var(--text-secondary);
        margin-right: var(--space-1);
    }

    .param {
        display: flex;
        align-items: center;
        gap: 2px;
    }

    .key {
        color: var(--text-muted);
    }

    .key::after {
        content: ":";
    }

    .val {
        color: var(--text-secondary);
    }

    .val.muted {
        color: var(--text-muted);
    }

    .token {
        padding: 0 3px;
        background: var(--bg-inset);
        border: 1px solid var(--border-subtle);
        white-space: pre;
    }

    .ce-target {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        padding-left: var(--space-2);
        border-left: 1px solid var(--border-subtle);
        margin-left: var(--space-1);
    }

    .target-text {
        color: var(--text-muted);
    }

    .label-token {
        background: var(--accent-primary-dim);
        border-color: var(--accent-primary);
        color: var(--accent-primary-bright);
    }
</style>
