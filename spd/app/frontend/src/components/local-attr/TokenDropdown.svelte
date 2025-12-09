<script lang="ts">
    import type { TokenInfo } from "../../lib/localAttributionsTypes";

    type Props = {
        tokens: TokenInfo[];
        value: string;
        onSelect: (tokenId: number, tokenString: string) => void;
        placeholder?: string;
    };

    let { tokens, value, onSelect, placeholder = "Search tokens..." }: Props = $props();

    let inputValue = $state(value);
    let isOpen = $state(false);
    let highlightedIndex = $state(0);

    // Sync external value changes
    $effect(() => {
        inputValue = value;
    });

    const filteredTokens = $derived.by(() => {
        if (!inputValue.trim()) return [];
        const search = inputValue.toLowerCase();
        const matches: TokenInfo[] = [];
        for (const t of tokens) {
            if (t.string.toLowerCase().includes(search)) {
                matches.push(t);
                if (matches.length >= 10) break;
            }
        }
        return matches;
    });

    function handleSelect(token: TokenInfo) {
        inputValue = token.string;
        onSelect(token.id, token.string);
        isOpen = false;
    }

    function handleKeydown(e: KeyboardEvent) {
        if (!isOpen || filteredTokens.length === 0) {
            if (e.key === "ArrowDown" && inputValue.trim()) {
                e.preventDefault();
                isOpen = true;
            }
            return;
        }

        switch (e.key) {
            case "ArrowDown":
                e.preventDefault();
                highlightedIndex = Math.min(highlightedIndex + 1, filteredTokens.length - 1);
                break;
            case "ArrowUp":
                e.preventDefault();
                highlightedIndex = Math.max(highlightedIndex - 1, 0);
                break;
            case "Enter":
                e.preventDefault();
                if (filteredTokens[highlightedIndex]) {
                    handleSelect(filteredTokens[highlightedIndex]);
                }
                break;
            case "Escape":
                e.preventDefault();
                isOpen = false;
                break;
        }
    }

    function handleInput() {
        isOpen = true;
        highlightedIndex = 0;
    }

    function handleFocus() {
        if (inputValue.trim()) {
            isOpen = true;
        }
    }

    function handleBackdropClick() {
        isOpen = false;
    }
</script>

<div class="token-dropdown">
    <input
        type="text"
        bind:value={inputValue}
        onfocus={handleFocus}
        onkeydown={handleKeydown}
        oninput={handleInput}
        {placeholder}
        class="dropdown-input"
    />

    {#if isOpen && filteredTokens.length > 0}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="dropdown-backdrop" onclick={handleBackdropClick} onkeydown={(e) => e.key === "Escape" && handleBackdropClick()}></div>
        <ul class="dropdown-list">
            {#each filteredTokens as token, i (token.id)}
                <li>
                    <button
                        type="button"
                        class="dropdown-item"
                        class:highlighted={i === highlightedIndex}
                        onclick={() => handleSelect(token)}
                        onmouseenter={() => (highlightedIndex = i)}
                    >
                        <span class="token-string">{token.string}</span>
                        <span class="token-id">#{token.id}</span>
                    </button>
                </li>
            {/each}
        </ul>
    {:else if isOpen && inputValue.trim() && filteredTokens.length === 0}
        <div class="dropdown-empty">No matching tokens</div>
    {/if}
</div>

<style>
    .token-dropdown {
        position: relative;
        display: inline-block;
    }

    .dropdown-input {
        width: 100px;
        padding: var(--space-1);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .dropdown-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .dropdown-input::placeholder {
        color: var(--text-muted);
    }

    .dropdown-backdrop {
        position: fixed;
        inset: 0;
        z-index: 999;
    }

    .dropdown-list {
        position: absolute;
        top: calc(100% + 2px);
        left: 0;
        min-width: 200px;
        max-height: 300px;
        overflow-y: auto;
        margin: 0;
        padding: 0;
        list-style: none;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 1000;
    }

    .dropdown-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        padding: var(--space-2) var(--space-3);
        background: transparent;
        border: none;
        cursor: pointer;
        text-align: left;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
    }

    .dropdown-item:hover,
    .dropdown-item.highlighted {
        background: var(--bg-inset);
    }

    .token-string {
        white-space: pre;
    }

    .token-id {
        font-size: var(--text-xs);
        color: var(--text-muted);
        margin-left: var(--space-2);
    }

    .dropdown-empty {
        position: absolute;
        top: calc(100% + 2px);
        left: 0;
        min-width: 150px;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        color: var(--text-muted);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        z-index: 1000;
    }
</style>
