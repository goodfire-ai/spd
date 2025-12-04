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
        padding: 0.4rem 0.75rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        white-space: nowrap;
    }

    .btn-add-prompt:hover {
        background: #1976d2;
    }

    .prompt-picker {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 0.5rem;
        width: 320px;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        z-index: 1000;
        overflow: hidden;
    }

    .picker-header {
        display: flex;
        gap: 0.5rem;
        padding: 0.75rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .picker-input {
        flex: 1;
        padding: 0.5rem 0.75rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    .picker-input:focus {
        outline: none;
        border-color: #2196f3;
    }

    .btn-tokenize {
        padding: 0.5rem 0.75rem;
        background: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 0.85rem;
        cursor: pointer;
    }

    .btn-tokenize:hover:not(:disabled) {
        background: #1976d2;
    }

    .btn-tokenize:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
    }

    .picker-filter {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: #fafafa;
        border-bottom: 1px solid #e0e0e0;
    }

    .filter-checkbox {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.8rem;
        color: #616161;
        cursor: pointer;
    }

    .filter-loading {
        font-size: 0.75rem;
        color: #999;
    }

    .picker-list {
        max-height: 240px;
        overflow-y: auto;
    }

    .picker-item {
        width: 100%;
        padding: 0.5rem 0.75rem;
        background: transparent;
        border: none;
        border-bottom: 1px solid #f0f0f0;
        cursor: pointer;
        text-align: left;
        display: flex;
        gap: 0.5rem;
        align-items: baseline;
    }

    .picker-item:hover {
        background: #f5f5f5;
    }

    .picker-item-id {
        font-size: 0.7rem;
        color: #9e9e9e;
        flex-shrink: 0;
    }

    .picker-item-preview {
        font-family: "SF Mono", Monaco, monospace;
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #424242;
    }

    .picker-empty {
        padding: 1.5rem;
        text-align: center;
        color: #9e9e9e;
        font-size: 0.85rem;
    }

    .picker-footer {
        padding: 0.5rem 0.75rem;
        border-top: 1px solid #e0e0e0;
        background: #fafafa;
    }

    .btn-generate {
        width: 100%;
        padding: 0.5rem;
        background: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
    }

    .btn-generate:hover {
        background: #43a047;
    }

    .generate-progress {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .mini-progress-bar {
        flex: 1;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
    }

    .mini-progress-fill {
        height: 100%;
        background: #4caf50;
        transition: width 0.1s ease;
    }
</style>
