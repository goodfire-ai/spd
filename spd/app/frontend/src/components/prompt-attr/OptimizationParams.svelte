<script lang="ts">
    import type { OptimizationResult } from "../../lib/promptAttributionsTypes";

    type Props = {
        optimization: OptimizationResult;
        tokens: string[];
    };

    let { optimization, tokens }: Props = $props();

    const config = $derived({
        steps: optimization.steps,
        impMinCoeff: optimization.imp_min_coeff,
        ceLossCoeff: optimization.ce_loss_coeff ?? 0,
        klLossCoeff: optimization.kl_loss_coeff ?? 0,
        lossSeqPos: optimization.loss_seq_pos,
        labelTokenText: optimization.label_str,
    });
</script>

<div class="opt-params">
    <span class="param"><span class="key">steps</span>{config.steps}</span>
    <span class="param"><span class="key">imp_min</span>{config.impMinCoeff}</span>
    {#if config.klLossCoeff > 0}
        <span class="param"><span class="key">kl</span>{config.klLossCoeff}</span>
    {/if}
    {#if config.ceLossCoeff > 0}
        <span class="param"><span class="key">ce</span>{config.ceLossCoeff}</span>
    {/if}
    <span class="param">
        <span class="key">pos</span>{config.lossSeqPos}{#if config.lossSeqPos >= 0 && config.lossSeqPos < tokens.length}
            (<span class="token">{tokens[config.lossSeqPos]}</span>){/if}
    </span>
    <span class="param">
        <span class="key">label</span>(<span class="token">{config.labelTokenText}</span>)
    </span>
</div>

<style>
    .opt-params {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-4);
        align-items: center;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-secondary);
    }

    .param {
        display: flex;
        align-items: center;
        gap: var(--space-1);
    }

    .key {
        color: var(--text-muted);
    }

    .key::after {
        content: ":";
    }

    .token {
        white-space: pre;
        font-family: var(--font-mono);
        background: var(--bg-inset);
        padding: 0 var(--space-1);
        border-radius: var(--radius-sm);
    }
</style>
