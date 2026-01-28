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

    // Get the current loss coefficient based on active loss type
    const currentLossCoeff = $derived(activeLossType === "kl" ? config.klLossCoeff : config.ceLossCoeff);

    // Slider value 0-100 (0 = sparsity "1000:1", 50 = balanced "1:1", 100 = reconstruction "1:1000")
    // Uses logarithmic scale: ratio = impMinCoeff / lossCoeff = 1000^(1 - value/50)
    const sliderValue = $derived.by(() => {
        const lossCoeff = Math.max(currentLossCoeff, 0.001); // avoid division by zero
        const ratio = config.impMinCoeff / lossCoeff;
        // ratio = 1000^(1 - value/50), so value = 50 * (1 - log10(ratio) / 3)
        const value = 50 * (1 - Math.log10(ratio) / 3);
        return Math.round(Math.max(0, Math.min(100, value)));
    });

    function handleSliderChange(value: number) {
        // ratio = 1000^(1 - value/50)
        const ratio = Math.pow(1000, 1 - value / 50);

        let impMinCoeff: number;
        let lossCoeff: number;

        if (ratio >= 1) {
            // Left half: impMinCoeff varies, lossCoeff = 1
            impMinCoeff = ratio;
            lossCoeff = 1;
        } else {
            // Right half: impMinCoeff = 1, lossCoeff varies
            impMinCoeff = 1;
            lossCoeff = 1 / ratio;
        }

        const update: Partial<OptimizeConfig> = { impMinCoeff };
        if (activeLossType === "kl") {
            update.klLossCoeff = lossCoeff;
        } else {
            update.ceLossCoeff = lossCoeff;
        }
        onChange(update);
    }

    function handleLossTypeChange(newType: LossType) {
        // Preserve the current loss coefficient when switching types
        const lossCoeff = Math.max(currentLossCoeff, 1);

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

    <div class="target-section">
        <div class="target-row">
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

    <div class="slider-section">
        <div class="slider-labels">
            <span class="slider-label">sparsity</span>
            <span class="slider-label">reconstruction</span>
        </div>
        <input
            type="range"
            class="balance-slider"
            min={0}
            max={100}
            value={sliderValue}
            oninput={(e) => handleSliderChange(parseInt(e.currentTarget.value))}
        />
        <div class="slider-labels">
            <span class="slider-label">1000:1</span>
            <span class="slider-label">1:1000</span>
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

    .target-section {
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .target-row {
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
        font-family: var(--font-mono);
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
</style>
