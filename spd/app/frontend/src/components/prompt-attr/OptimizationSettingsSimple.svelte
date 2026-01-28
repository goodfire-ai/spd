<script lang="ts">
    import type { OptimizeConfig } from "./types";
    import type { TokenInfo } from "../../lib/promptAttributionsTypes";
    import TokenDropdown from "./TokenDropdown.svelte";

    type LossType = "kl" | "ce";

    type Props = {
        config: OptimizeConfig;
        tokens: string[];
        allTokens: TokenInfo[];
        onChange: (partial: Partial<OptimizeConfig>) => void;
    };

    let { config, tokens, allTokens, onChange }: Props = $props();

    // Determine which loss type is active based on coefficients
    // CE mode is active when ceLossCoeff > 0 (labelTokenId validation is for compute time)
    const activeLossType = $derived.by((): LossType => {
        if (config.ceLossCoeff > 0) return "ce";
        return "kl";
    });

    // Slider value 0-100 (0 = sparsity, 100 = reconstruction)
    // Derived from impMinCoeff: 1.0 -> 0, 0.01 -> 100
    const sliderValue = $derived.by(() => {
        // impMinCoeff ranges from 1.0 (slider=0) to 0.01 (slider=100)
        // t = (1.0 - impMinCoeff) / (1.0 - 0.01)
        const t = (1.0 - config.impMinCoeff) / 0.99;
        return Math.round(t * 100);
    });

    function handleSliderChange(value: number) {
        const t = value / 100;
        // impMinCoeff: 1.0 at t=0, 0.01 at t=1
        const impMinCoeff = 1.0 * (1 - t) + 0.01 * t;
        // lossCoeff: 0.1 at t=0, 1.0 at t=1
        const lossCoeff = 0.1 * (1 - t) + 1.0 * t;

        const update: Partial<OptimizeConfig> = { impMinCoeff };
        if (activeLossType === "kl") {
            update.klLossCoeff = lossCoeff;
        } else {
            update.ceLossCoeff = lossCoeff;
        }
        onChange(update);
    }

    function handleLossTypeChange(newType: LossType) {
        // Calculate current lossCoeff from slider
        const t = sliderValue / 100;
        const lossCoeff = 0.1 * (1 - t) + 1.0 * t;

        if (newType === "kl") {
            onChange({
                klLossCoeff: lossCoeff,
                ceLossCoeff: 0,
            });
        } else {
            onChange({
                klLossCoeff: 0,
                ceLossCoeff: lossCoeff,
            });
        }
    }

    const tokenAtSeqPos = $derived(
        config.lossSeqPos >= 0 && config.lossSeqPos < tokens.length ? tokens[config.lossSeqPos] : null,
    );
</script>

<div class="simple-settings">
    <div class="loss-type-section">
        <div class="section-label">Loss Type</div>
        <div class="loss-type-options">
            <label class="loss-type-option" class:selected={activeLossType === "kl"}>
                <input
                    type="radio"
                    name="loss-type"
                    checked={activeLossType === "kl"}
                    onchange={() => handleLossTypeChange("kl")}
                />
                <div class="option-content">
                    <span class="option-name">KL Divergence</span>
                    <span class="option-desc">match full output distribution</span>
                </div>
            </label>
            <label class="loss-type-option" class:selected={activeLossType === "ce"}>
                <input
                    type="radio"
                    name="loss-type"
                    checked={activeLossType === "ce"}
                    onchange={() => handleLossTypeChange("ce")}
                />
                <div class="option-content">
                    <span class="option-name">Cross-Entropy</span>
                    <span class="option-desc">predict specific token</span>
                </div>
            </label>
        </div>
    </div>

    {#if activeLossType === "ce"}
        <div class="ce-target-section">
            <div class="ce-target-row">
                <span class="target-label">At position</span>
                <input
                    type="number"
                    class="pos-input"
                    value={config.lossSeqPos}
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
                <div class="label-input">
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
                </div>
            </div>
        </div>
    {/if}

    <div class="slider-section">
        <div class="slider-labels">
            <span class="slider-label">Sparsity</span>
            <span class="slider-label">Reconstruction</span>
        </div>
        <input
            type="range"
            class="balance-slider"
            min={0}
            max={100}
            value={sliderValue}
            oninput={(e) => handleSliderChange(parseInt(e.currentTarget.value))}
        />
        <div class="slider-value">
            <span class="value-label">imp_min: {config.impMinCoeff.toFixed(2)}</span>
            <span class="value-label">
                {activeLossType === "kl" ? "kl" : "ce"}_coeff: {(activeLossType === "kl"
                    ? config.klLossCoeff
                    : config.ceLossCoeff
                ).toFixed(2)}
            </span>
        </div>
    </div>
</div>

<style>
    .simple-settings {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        max-width: 400px;
    }

    .section-label {
        font-size: var(--text-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: var(--space-1);
    }

    .loss-type-options {
        display: flex;
        gap: var(--space-2);
    }

    .loss-type-option {
        flex: 1;
        display: flex;
        align-items: flex-start;
        gap: var(--space-2);
        padding: var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        cursor: pointer;
        transition: border-color 0.15s;
    }

    .loss-type-option:hover {
        border-color: var(--border-strong);
    }

    .loss-type-option.selected {
        border-color: var(--accent-primary);
        background: var(--bg-surface);
    }

    .loss-type-option input {
        margin-top: 2px;
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .option-content {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .option-name {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .option-desc {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .ce-target-section {
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px dashed var(--border-default);
    }

    .ce-target-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
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

    .slider-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .slider-labels {
        display: flex;
        justify-content: space-between;
    }

    .slider-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .balance-slider {
        width: 100%;
        height: 20px;
        appearance: none;
        background: transparent;
        cursor: pointer;
    }

    .balance-slider::-webkit-slider-runnable-track {
        width: 100%;
        height: 6px;
        background: var(--border-default);
        border-radius: 3px;
    }

    .balance-slider::-webkit-slider-thumb {
        appearance: none;
        width: 16px;
        height: 16px;
        background: var(--accent-primary);
        border-radius: 50%;
        cursor: pointer;
        margin-top: -5px;
    }

    .balance-slider::-moz-range-track {
        width: 100%;
        height: 6px;
        background: var(--border-default);
        border-radius: 3px;
    }

    .balance-slider::-moz-range-thumb {
        width: 16px;
        height: 16px;
        background: var(--accent-primary);
        border-radius: 50%;
        cursor: pointer;
        border: none;
    }

    .slider-value {
        display: flex;
        justify-content: space-between;
    }

    .value-label {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }
</style>
