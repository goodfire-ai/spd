<script lang="ts">
    import type { PromptPreview, PinnedNode } from "../../lib/localAttributionsTypes";

    type Props = {
        prompts: PromptPreview[];
        filteredPrompts: PromptPreview[];
        pinnedNodes: PinnedNode[];
        filterByPinned: boolean;
        filterLoading: boolean;
        generatingGraphs: boolean;
        generateProgress: number;
        generateCount: number;
        onSelectPrompt: (prompt: PromptPreview) => void;
        onAddCustom: (text: string) => Promise<void>;
        onFilterToggle: () => void;
        onGenerate: (count: number) => void;
    };

    let {
        prompts,
        filteredPrompts,
        pinnedNodes,
        filterByPinned,
        filterLoading,
        generatingGraphs,
        generateProgress,
        generateCount,
        onSelectPrompt,
        onAddCustom,
        onFilterToggle,
        onGenerate,
    }: Props = $props();

    let showPicker = $state(false);
    let customText = $state("");
    let tokenizeLoading = $state(false);

    const displayedPrompts = $derived(filterByPinned ? filteredPrompts : prompts);

    async function handleAddCustom() {
        if (!customText.trim() || tokenizeLoading) return;
        tokenizeLoading = true;
        try {
            await onAddCustom(customText);
            customText = "";
            showPicker = false;
        } finally {
            tokenizeLoading = false;
        }
    }
</script>

<div class="add-prompt-wrapper">
    <button class="btn-add-prompt" onclick={() => (showPicker = !showPicker)}> + Add Prompt </button>

    {#if showPicker}
        <div class="prompt-picker">
            <div class="picker-header">
                <input
                    type="text"
                    bind:value={customText}
                    placeholder="Enter custom text..."
                    onkeydown={(e) => e.key === "Enter" && handleAddCustom()}
                    class="picker-input"
                />
                <button onclick={handleAddCustom} disabled={!customText.trim() || tokenizeLoading} class="btn-tokenize">
                    {tokenizeLoading ? "..." : "Add"}
                </button>
            </div>

            <div class="picker-filter">
                <label class="filter-checkbox">
                    <input
                        type="checkbox"
                        checked={filterByPinned}
                        onchange={onFilterToggle}
                        disabled={pinnedNodes.length === 0}
                    />
                    Filter by pinned ({pinnedNodes.length})
                </label>
                {#if filterLoading}
                    <span class="filter-loading">...</span>
                {/if}
            </div>

            <div class="picker-list">
                {#each displayedPrompts as p (p.id)}
                    <button
                        class="picker-item"
                        onclick={() => {
                            onSelectPrompt(p);
                            showPicker = false;
                        }}
                    >
                        <span class="picker-item-id">#{p.id}</span>
                        <span class="picker-item-preview">{p.preview}</span>
                    </button>
                {/each}
                {#if displayedPrompts.length === 0}
                    <div class="picker-empty">
                        {filterByPinned ? "No matching prompts" : "No prompts yet"}
                    </div>
                {/if}
            </div>

            <div class="picker-footer">
                {#if generatingGraphs}
                    <div class="generate-progress">
                        <div class="mini-progress-bar">
                            <div class="mini-progress-fill" style="width: {generateProgress * 100}%"></div>
                        </div>
                        <span>{generateCount}</span>
                    </div>
                {:else}
                    <button class="btn-generate" onclick={() => onGenerate(100)}> + Generate 100 </button>
                {/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .add-prompt-wrapper {
        position: relative;
    }

    .btn-add-prompt {
        padding: var(--space-1) var(--space-3);
        background: var(--accent-amber);
        color: var(--bg-base);
        border: none;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .btn-add-prompt:hover {
        background: var(--text-primary);
    }

    .prompt-picker {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: var(--space-2);
        width: 340px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        z-index: 1000;
        overflow: hidden;
    }

    .picker-header {
        display: flex;
        gap: var(--space-2);
        padding: var(--space-3);
        border-bottom: 1px solid var(--border-default);
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

    .picker-input:focus {
        outline: none;
        border-color: var(--accent-amber-dim);
    }

    .picker-input::placeholder {
        color: var(--text-muted);
    }

    .btn-tokenize {
        padding: var(--space-2) var(--space-3);
        background: var(--accent-amber);
        color: var(--bg-base);
        border: none;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 600;
        cursor: pointer;
        text-transform: uppercase;
    }

    .btn-tokenize:hover:not(:disabled) {
        background: var(--text-primary);
    }

    .btn-tokenize:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
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
        color: var(--text-primary);
        border: none;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 600;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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
</style>
