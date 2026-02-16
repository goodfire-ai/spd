<script lang="ts">
    import type { OptimizeConfigDraft, MaskType, LossType, LossConfigDraft } from "./types";
    import TokenDropdown from "./TokenDropdown.svelte";

    type Props = {
        config: OptimizeConfigDraft;
        tokens: string[];
        onChange: (newConfig: OptimizeConfigDraft) => void;
        cardId: number;
    };

    let { config, tokens, onChange, cardId }: Props = $props();
    let showAdvanced = $state(false);

    // Slider value 0-100 controls impMinCoeff on log scale from 1e-5 to 10
    const sliderValue = $derived.by(() => {
        const value = (100 * Math.log10(config.impMinCoeff / 1e-5)) / 6;
        return Math.round(Math.max(0, Math.min(100, value)));
    });

    function handleSliderChange(value: number) {
        const impMinCoeff = 1e-5 * Math.pow(1e6, value / 100);
        const rounded = parseFloat(impMinCoeff.toPrecision(2));
        onChange({ ...config, impMinCoeff: rounded });
    }

    function handleLossTypeChange(newType: LossType) {
        const position = config.loss.position;
        const coeff = config.loss.coeff;
        let newLoss: LossConfigDraft;
        if (newType === "kl") {
            newLoss = { type: "kl", coeff, position };
        } else {
            newLoss = {
                type: "ce",
                coeff,
                position,
                labelTokenId: null,
                labelTokenText: "",
            };
        }
        onChange({ ...config, loss: newLoss });
    }

    const tokenAtSeqPos = $derived(
        config.loss.position >= 0 && config.loss.position < tokens.length ? tokens[config.loss.position] : null,
    );
</script>

<div class="opt-settings">
    <!-- Loss type selection -->
    <div class="loss-type-options">
        <label class="loss-type-option" class:selected={config.loss.type === "kl"}>
            <input
                type="radio"
                name="loss-type-{cardId}"
                checked={config.loss.type === "kl"}
                onchange={() => handleLossTypeChange("kl")}
            />
            <span class="option-name">KL Divergence</span>
        </label>
        <label class="loss-type-option" class:selected={config.loss.type === "ce"}>
            <input
                type="radio"
                name="loss-type-{cardId}"
                checked={config.loss.type === "ce"}
                onchange={() => handleLossTypeChange("ce")}
            />
            <span class="option-name">Cross-Entropy</span>
        </label>
    </div>

    <!-- Target position and token -->
    <div class="target-section">
        <span class="target-label">At position</span>
        <input
            type="number"
            class="pos-input"
            value={config.loss.position}
            oninput={(e) => {
                if (e.currentTarget.value === "") return;
                const position = parseInt(e.currentTarget.value);
                onChange({ ...config, loss: { ...config.loss, position } });
            }}
            min={0}
            max={tokens.length - 1}
            step={1}
        />
        {#if tokenAtSeqPos !== null}
            (<span class="token">{tokenAtSeqPos}</span>)
        {/if}
        {#if config.loss.type === "ce"}
            <span class="target-label">, predict</span>
            <TokenDropdown
                value={config.loss.labelTokenText}
                selectedTokenId={config.loss.labelTokenId}
                onSelect={(tokenId, tokenString) => {
                    if (config.loss.type !== "ce")
                        throw new Error(
                            "inconsistent state: Token dropdown rendered but loss not type CE but no label token",
                        );

                    if (tokenId !== null) {
                        onChange({
                            ...config,
                            loss: { ...config.loss, labelTokenId: tokenId, labelTokenText: tokenString },
                        });
                    }
                }}
                placeholder="token..."
            />
        {/if}
    </div>

    <!-- Sparsity slider -->
    <div class="slider-section">
        <div class="slider-header">
            <span class="section-label">Sparsity</span>
            <input
                type="text"
                class="imp-min-input"
                value={config.impMinCoeff.toPrecision(2)}
                onchange={(e) => {
                    const val = parseFloat(e.currentTarget.value);
                    if (!isNaN(val) && val > 0) {
                        onChange({ ...config, impMinCoeff: val });
                    }
                }}
            />
        </div>
        <input
            type="range"
            class="sparsity-slider"
            min={0}
            max={100}
            value={sliderValue}
            oninput={(e) => handleSliderChange(parseInt(e.currentTarget.value))}
        />
        <div class="slider-labels">
            <span class="slider-label">1e-5</span>
            <span class="slider-label">10</span>
        </div>
    </div>

    <!-- Advanced toggle -->
    <button class="advanced-toggle" onclick={() => (showAdvanced = !showAdvanced)}>
        {showAdvanced ? "▼" : "▶"} Advanced
    </button>

    {#if showAdvanced}
        <div class="advanced-section">
            <div class="settings-grid">
                <label>
                    <span class="label-text">steps</span>
                    <input
                        type="number"
                        value={config.steps}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ ...config, steps: parseInt(e.currentTarget.value) });
                        }}
                        min={10}
                        max={5000}
                        step={100}
                    />
                </label>
                <label>
                    <span class="label-text">pnorm</span>
                    <input
                        type="number"
                        value={config.pnorm}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ ...config, pnorm: parseFloat(e.currentTarget.value) });
                        }}
                        min={0.1}
                        max={2}
                        step={0.1}
                    />
                </label>
                <label>
                    <span class="label-text">beta</span>
                    <input
                        type="number"
                        value={config.beta}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ ...config, beta: parseFloat(e.currentTarget.value) });
                        }}
                        min={0}
                        max={10}
                        step={0.1}
                    />
                </label>
                <label>
                    <span class="label-text">mask_type</span>
                    <select
                        value={config.maskType}
                        onchange={(e) => onChange({ ...config, maskType: e.currentTarget.value as MaskType })}
                    >
                        <option value="stochastic">stochastic</option>
                        <option value="ci">ci</option>
                    </select>
                </label>
                <label>
                    <span class="label-text">loss_coeff</span>
                    <input
                        type="number"
                        value={config.loss.coeff}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            const coeff = parseFloat(e.currentTarget.value);
                            onChange({ ...config, loss: { ...config.loss, coeff } });
                        }}
                        min={0}
                        step={0.1}
                    />
                </label>
                <label>
                    <span class="label-text">adv_n_steps</span>
                    <input
                        type="number"
                        value={config.advPgdNSteps ?? ""}
                        oninput={(e) => {
                            const val = e.currentTarget.value;
                            onChange({ ...config, advPgdNSteps: val === "" ? null : parseInt(val) });
                        }}
                        min={1}
                        max={50}
                        step={1}
                        placeholder="off"
                    />
                </label>
                <label>
                    <span class="label-text">adv_step_size</span>
                    <input
                        type="number"
                        value={config.advPgdStepSize ?? ""}
                        oninput={(e) => {
                            const val = e.currentTarget.value;
                            onChange({ ...config, advPgdStepSize: val === "" ? null : parseFloat(val) });
                        }}
                        min={0.001}
                        max={1}
                        step={0.01}
                        placeholder="off"
                    />
                </label>
            </div>
        </div>
    {/if}
</div>

<style>
    .opt-settings {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        max-width: 400px;
    }

    .loss-type-options {
        display: flex;
        gap: var(--space-2);
    }

    .loss-type-option {
        flex: 1;
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        cursor: pointer;
        transition: border-color var(--transition-normal);
    }

    .loss-type-option:hover {
        border-color: var(--border-strong);
    }

    .loss-type-option.selected {
        border-color: var(--accent-primary);
        background: var(--bg-surface);
    }

    .loss-type-option input {
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .option-name {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .target-section {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
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

    .token {
        white-space: pre;
        font-family: var(--font-mono);
        background: var(--bg-inset);
        padding: 0 var(--space-1);
        border-radius: var(--radius-sm);
    }

    .slider-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .slider-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .section-label {
        font-size: var(--text-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    .imp-min-input {
        width: 80px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .imp-min-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
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

    .sparsity-slider {
        width: 100%;
        height: 20px;
        appearance: none;
        background: transparent;
        cursor: pointer;
    }

    .sparsity-slider::-webkit-slider-runnable-track {
        width: 100%;
        height: 6px;
        background: var(--border-default);
        border-radius: var(--radius-sm);
    }

    .sparsity-slider::-webkit-slider-thumb {
        appearance: none;
        width: 16px;
        height: 16px;
        background: var(--accent-primary);
        border-radius: 50%;
        cursor: pointer;
        margin-top: -5px;
    }

    .sparsity-slider::-moz-range-track {
        width: 100%;
        height: 6px;
        background: var(--border-default);
        border-radius: var(--radius-sm);
    }

    .sparsity-slider::-moz-range-thumb {
        width: 16px;
        height: 16px;
        background: var(--accent-primary);
        border-radius: 50%;
        cursor: pointer;
        border: none;
    }

    .advanced-toggle {
        background: none;
        border: none;
        padding: var(--space-1) 0;
        font-size: var(--text-xs);
        color: var(--text-muted);
        cursor: pointer;
        text-align: left;
    }

    .advanced-toggle:hover {
        color: var(--text-secondary);
    }

    .advanced-section {
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .settings-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        gap: var(--space-2);
    }

    .settings-grid label {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .label-text {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
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
</style>
