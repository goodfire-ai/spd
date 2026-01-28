<script lang="ts">
    import type { OptimizeConfig, MaskType } from "./types";
    import type { TokenInfo } from "../../lib/promptAttributionsTypes";
    import TokenDropdown from "./TokenDropdown.svelte";

    type Props = {
        config: OptimizeConfig;
        tokens: string[];
        allTokens: TokenInfo[];
        onChange: (partial: Partial<OptimizeConfig>) => void;
    };

    let { config, tokens, allTokens, onChange }: Props = $props();

    const tokenAtSeqPos = $derived(
        config.lossSeqPos >= 0 && config.lossSeqPos < tokens.length ? tokens[config.lossSeqPos] : null,
    );

    const descriptions = {
        impMinCoeff:
            "Importance minimality coefficient. Higher values penalize components for being active, encouraging sparser circuits.",
        steps: "Number of optimization steps. More steps allow for better convergence but take longer.",
        pnorm: "P-norm exponent for importance minimality. Lower values (e.g. 0.3) encourage sparser solutions than L1.",
        beta: "Temperature for stochastic mask sampling. Higher values make masks more deterministic.",
        ceLossCoeff: "Cross-entropy loss coefficient. When > 0, penalizes circuits that don't predict the label token.",
        klLossCoeff:
            "KL divergence loss coefficient. When > 0, penalizes circuits that deviate from the original output distribution.",
        lossSeqPos:
            "Token position where the prediction loss is computed. The model predicts the next token at this position.",
        labelToken: "The target token the model should predict at the loss position. Required when using CE loss.",
        maskType:
            "How to apply component masks. 'stochastic' samples binary masks from CI values; 'ci' uses CI values directly as soft masks.",
    };
</script>

<div class="opt-settings">
    <div class="settings-section">
        <div class="section-header">Optimization</div>
        <div class="settings-grid">
            <label title={descriptions.steps}>
                <span class="label-text">steps</span>
                <input
                    type="number"
                    value={config.steps}
                    oninput={(e) => {
                        if (e.currentTarget.value === "") return;
                        onChange({ steps: parseInt(e.currentTarget.value) });
                    }}
                    min={10}
                    max={5000}
                    step={100}
                />
            </label>
            <label title={descriptions.impMinCoeff}>
                <span class="label-text">imp_min_coeff</span>
                <input
                    type="number"
                    value={config.impMinCoeff}
                    oninput={(e) => {
                        if (e.currentTarget.value === "") return;
                        onChange({ impMinCoeff: parseFloat(e.currentTarget.value) });
                    }}
                    min={0.001}
                    max={10}
                    step={0.01}
                />
            </label>
            <label title={descriptions.pnorm}>
                <span class="label-text">pnorm</span>
                <input
                    type="number"
                    value={config.pnorm}
                    oninput={(e) => {
                        if (e.currentTarget.value === "") return;
                        onChange({ pnorm: parseFloat(e.currentTarget.value) });
                    }}
                    min={0.1}
                    max={2}
                    step={0.1}
                />
            </label>
            <label title={descriptions.beta}>
                <span class="label-text">beta</span>
                <input
                    type="number"
                    value={config.beta}
                    oninput={(e) => {
                        if (e.currentTarget.value === "") return;
                        onChange({ beta: parseFloat(e.currentTarget.value) });
                    }}
                    min={0}
                    max={10}
                    step={0.1}
                />
            </label>
            <label title={descriptions.maskType}>
                <span class="label-text">mask_type</span>
                <select
                    value={config.maskType}
                    onchange={(e) => onChange({ maskType: e.currentTarget.value as MaskType })}
                >
                    <option value="stochastic">stochastic</option>
                    <option value="ci">ci</option>
                </select>
            </label>
        </div>
    </div>

    <div class="settings-section loss-section">
        <div class="section-header">Loss Function</div>

        <div class="loss-options">
            <div class="loss-option" title={descriptions.klLossCoeff}>
                <div class="loss-option-header">
                    <span class="loss-name">KL Divergence</span>
                    <span class="loss-desc">match original output distribution</span>
                </div>
                <label class="coeff-row">
                    <span class="label-text">coeff</span>
                    <input
                        type="number"
                        value={config.klLossCoeff}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ klLossCoeff: parseFloat(e.currentTarget.value) });
                        }}
                        min={0}
                        step={0.1}
                    />
                </label>
            </div>

            <div class="loss-option ce-option" title={descriptions.ceLossCoeff}>
                <div class="loss-option-header">
                    <span class="loss-name">Cross-Entropy</span>
                    <span class="loss-desc">predict specific token at position</span>
                </div>
                <div class="ce-fields">
                    <label class="coeff-row">
                        <span class="label-text">coeff</span>
                        <input
                            type="number"
                            value={config.ceLossCoeff}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onChange({ ceLossCoeff: parseFloat(e.currentTarget.value) });
                            }}
                            min={0}
                            step={0.1}
                        />
                    </label>
                    <div class="ce-target">
                        <span class="target-label">At position</span>
                        <input
                            type="number"
                            class="pos-input"
                            value={config.lossSeqPos}
                            title={descriptions.lossSeqPos}
                            oninput={(e) => {
                                if (e.currentTarget.value === "") return;
                                onChange({ lossSeqPos: parseInt(e.currentTarget.value) });
                            }}
                            min={0}
                            max={tokens.length - 1}
                            step={1}
                        />
                        {#if tokenAtSeqPos !== null}
                            <span class="token-at-pos">{tokenAtSeqPos}</span>
                        {:else}
                            <span class="token-at-pos invalid">invalid</span>
                        {/if}
                        <span class="target-label">predict</span>
                        <div class="label-input" title={descriptions.labelToken}>
                            <TokenDropdown
                                tokens={allTokens}
                                value={config.labelTokenText}
                                selectedTokenId={config.labelTokenId}
                                onSelect={(tokenId, tokenString) => {
                                    onChange({
                                        labelTokenText: tokenString,
                                        labelTokenId: tokenId,
                                        labelTokenPreview: tokenId !== null ? tokenString : "",
                                    });
                                }}
                                placeholder="token..."
                            />
                            {#if config.labelTokenId !== null}
                                <span class="token-id-hint">#{config.labelTokenId}</span>
                            {/if}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    .opt-settings {
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
    }

    .settings-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .section-header {
        font-size: var(--text-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    .settings-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
        gap: var(--space-2);
    }

    .settings-grid label {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .label-text {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        font-weight: 500;
    }

    .settings-grid input[type="number"],
    .settings-grid select {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .settings-grid input[type="number"]:focus,
    .settings-grid select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .settings-grid select {
        cursor: pointer;
    }

    .loss-section {
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .loss-options {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .loss-option {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding: var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
    }

    .loss-option-header {
        display: flex;
        align-items: baseline;
        gap: var(--space-2);
    }

    .loss-name {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .loss-desc {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .coeff-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .coeff-row input[type="number"] {
        width: 70px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-base);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .coeff-row input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .ce-fields {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .ce-target {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
        padding: var(--space-2);
        background: var(--bg-inset);
        border: 1px dashed var(--border-subtle);
    }

    .target-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .pos-input {
        width: 50px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-base);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .pos-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .token-at-pos {
        padding: 2px 6px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-secondary);
        white-space: pre;
    }

    .token-at-pos.invalid {
        font-style: italic;
        color: var(--status-negative);
        border-color: var(--status-negative);
    }

    .label-input {
        display: flex;
        align-items: center;
        gap: var(--space-1);
    }

    .token-id-hint {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
</style>
