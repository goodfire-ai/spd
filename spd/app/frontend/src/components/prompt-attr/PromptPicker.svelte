<script lang="ts">
    import type { PromptPreview, PinnedNode, TokenizeResult } from "../../lib/promptAttributionsTypes";
    import { tokenizeText } from "../../lib/api";
    import type { PromptGenerateState } from "./types";

    type Props = {
        prompts: PromptPreview[];
        filteredPrompts: PromptPreview[];
        stagedNodes: PinnedNode[];
        filterByStaged: boolean;
        filterLoading: boolean;
        promptGenerate: PromptGenerateState;
        isAddingCustomPrompt: boolean;
        show: boolean;
        onSelectPrompt: (prompt: PromptPreview) => void;
        onAddCustom: (text: string) => Promise<void>;
        onFilterToggle: () => void;
        onGenerate: (count: number) => void;
        onClose: () => void;
    };

    let {
        prompts,
        filteredPrompts,
        stagedNodes,
        filterByStaged,
        filterLoading,
        promptGenerate,
        isAddingCustomPrompt,
        show,
        onSelectPrompt,
        onAddCustom,
        onFilterToggle,
        onGenerate,
        onClose,
    }: Props = $props();

    let customText = $state("");
    let tokenizeResult = $state<TokenizeResult | null>(null);
    let tokenizeLoading = $state(false);
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;

    async function runTokenize(text: string) {
        if (!text.trim()) {
            tokenizeResult = null;
            return;
        }
        tokenizeLoading = true;
        try {
            tokenizeResult = await tokenizeText(text);
        } finally {
            tokenizeLoading = false;
        }
    }

    function onCustomInput(e: Event) {
        const target = e.target as HTMLInputElement;
        customText = target.value;
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => runTokenize(customText), 150);
    }

    const displayedPrompts = $derived(filterByStaged ? filteredPrompts : prompts);

    async function handleAddCustom() {
        if (!customText.trim() || isAddingCustomPrompt) return;
        await onAddCustom(customText);
        customText = "";
        tokenizeResult = null;
        onClose();
    }

    function handleSelectPrompt(p: PromptPreview) {
        onSelectPrompt(p);
        onClose();
    }
</script>

{#if show}
    <div class="prompt-picker-backdrop" onclick={onClose} onkeydown={(e) => e.key === "Escape" && onClose()}></div>
    <div class="prompt-picker">
        <div class="picker-header">
            <div class="custom-input-section">
                <div class="input-row">
                    <input
                        type="text"
                        value={customText}
                        oninput={onCustomInput}
                        placeholder="Enter custom text..."
                        onkeydown={(e) => e.key === "Enter" && handleAddCustom()}
                        class="picker-input"
                    />
                    <button
                        onclick={handleAddCustom}
                        disabled={!customText.trim() || isAddingCustomPrompt}
                        class="btn-tokenize"
                    >
                        {isAddingCustomPrompt ? "..." : "Add"}
                    </button>
                </div>
                {#if tokenizeLoading}
                    <div class="token-preview-loading">...</div>
                {:else if tokenizeResult && tokenizeResult.tokens.length > 0}
                    <div class="token-preview">
                        {#each tokenizeResult.tokens as tok, i (i)}<span class="token">{tok}</span>{/each}
                        <span class="token-count">({tokenizeResult.tokens.length})</span>
                    </div>
                {/if}
            </div>
        </div>

        <div class="picker-filter">
            <label class="filter-checkbox">
                <input
                    type="checkbox"
                    checked={filterByStaged}
                    onchange={onFilterToggle}
                    disabled={stagedNodes.length === 0}
                />
                Filter by staged ({stagedNodes.length})
            </label>
            {#if filterLoading}
                <span class="filter-loading">...</span>
            {/if}
        </div>

        <div class="picker-list">
            {#each displayedPrompts as p (p.id)}
                <button class="picker-item" onclick={() => handleSelectPrompt(p)}>
                    <span class="picker-item-id">#{p.id}</span>
                    <span class="picker-item-preview">{p.preview}</span>
                </button>
            {/each}
            {#if displayedPrompts.length === 0}
                <div class="picker-empty">
                    {filterByStaged ? "No matching prompts" : "No prompts yet"}
                </div>
            {/if}
        </div>

        <div class="picker-footer">
            {#if promptGenerate.status === "generating"}
                <div class="generate-progress">
                    <div class="mini-progress-bar">
                        <div class="mini-progress-fill" style="width: {promptGenerate.progress * 100}%"></div>
                    </div>
                    <span>{promptGenerate.count}</span>
                </div>
            {:else if promptGenerate.status === "error"}
                <div class="generate-error">{promptGenerate.error}</div>
            {:else}
                <button class="btn-generate" onclick={() => onGenerate(50)}> + Generate 50 </button>
            {/if}
        </div>
    </div>
{/if}

<style>
    .prompt-picker-backdrop {
        position: fixed;
        inset: 0;
        z-index: 999;
    }

    .prompt-picker {
        position: absolute;
        top: calc(100% + var(--space-1));
        left: 0;
        width: 340px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        overflow: hidden;
    }

    .picker-header {
        padding: var(--space-3);
        border-bottom: 1px solid var(--border-default);
    }

    .custom-input-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .input-row {
        display: flex;
        gap: var(--space-2);
    }

    .picker-input {
        flex: 1;
        padding: var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .token-preview {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        align-items: center;
    }

    .token-preview-loading {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .token {
        padding: 2px 3px;
        background: var(--bg-inset);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--status-info-bright);
        white-space: pre;
        border: 1px solid var(--status-info);
    }

    .token-count {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
        margin-left: var(--space-1);
    }

    .picker-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .picker-input::placeholder {
        color: var(--text-muted);
    }

    .btn-tokenize {
        padding: var(--space-2) var(--space-3);
        background: var(--accent-primary);
        color: white;
        border: none;
        font-weight: 500;
    }

    .btn-tokenize:hover:not(:disabled) {
        background: var(--accent-primary-dim);
    }

    .btn-tokenize:disabled {
        background: var(--border-default);
        color: var(--text-muted);
    }

    .picker-filter {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
    }

    .filter-checkbox {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        cursor: pointer;
    }

    .filter-loading {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .picker-list {
        max-height: 260px;
        overflow-y: auto;
        background: var(--bg-inset);
    }

    .picker-item {
        width: 100%;
        padding: var(--space-2) var(--space-3);
        background: transparent;
        border: none;
        border-bottom: 1px solid var(--border-subtle);
        cursor: pointer;
        text-align: left;
        display: flex;
        gap: var(--space-2);
        align-items: baseline;
        color: var(--text-primary);
    }

    .picker-item:hover {
        background: var(--bg-surface);
    }

    .picker-item-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .picker-item-preview {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--text-primary);
    }

    .picker-empty {
        padding: var(--space-4);
        text-align: center;
        color: var(--text-muted);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
    }

    .picker-footer {
        padding: var(--space-2) var(--space-3);
        border-top: 1px solid var(--border-default);
        background: var(--bg-surface);
    }

    .btn-generate {
        width: 100%;
        padding: var(--space-2);
        background: var(--status-positive);
        color: white;
        border: none;
        font-weight: 500;
    }

    .btn-generate:hover {
        background: var(--status-positive-bright);
    }

    .generate-progress {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-secondary);
    }

    .mini-progress-bar {
        flex: 1;
        height: 4px;
        background: var(--border-default);
        overflow: hidden;
    }

    .mini-progress-fill {
        height: 100%;
        background: var(--status-positive);
        transition: width 0.1s ease;
    }

    .generate-error {
        font-size: var(--text-sm);
        color: var(--status-negative-bright);
        font-family: var(--font-mono);
    }
</style>
