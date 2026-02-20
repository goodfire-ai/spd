<script lang="ts">
    import type { PromptCard } from "./types";

    type Props = {
        cards: PromptCard[];
        activeCardId: number | null;
        onSelectCard: (cardId: number) => void;
        onCloseCard: (cardId: number) => void;
        onAddClick: () => void;
    };

    let { cards, activeCardId, onSelectCard, onCloseCard, onAddClick }: Props = $props();

    function getCardLabel(card: PromptCard): string {
        const nCharsToShow = 30;
        const str = card.tokens.join("");
        return str.slice(0, nCharsToShow) + (str.length > nCharsToShow ? "..." : "");
    }
</script>

<div class="card-tabs">
    {#each cards as card (card.id)}
        <div class="card-tab" class:active={card.id === activeCardId}>
            <button class="card-tab-label" onclick={() => onSelectCard(card.id)}>
                {getCardLabel(card)}
            </button>
            <button class="card-tab-close" onclick={() => onCloseCard(card.id)}>×</button>
        </div>
    {/each}
    <button class="btn-add-tab" onclick={onAddClick}>+{cards.length > 0 ? "" : " Add Prompt"}</button>
</div>

<style>
    .card-tabs {
        display: flex;
        gap: var(--space-2);
        flex: 1;
        overflow-x: auto;
    }

    .card-tab {
        display: flex;
        align-items: center;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .card-tab:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .card-tab.active {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
        color: var(--text-primary);
    }

    .card-tab-label {
        padding: var(--space-1) var(--space-2);
        background: transparent;
        border: none;
        font-size: inherit;
        font-family: inherit;
        color: inherit;
        cursor: pointer;
        max-width: 140px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .card-tab-close {
        padding: var(--space-1);
        background: transparent;
        border: none;
        border-left: 1px solid var(--border-subtle);
        font-size: var(--text-sm);
        line-height: 1;
        opacity: 0.5;
        cursor: pointer;
        color: inherit;
    }

    .card-tab-close:hover {
        opacity: 1;
        color: var(--status-negative-bright);
    }

    .btn-add-tab {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-surface);
        border: 1px dashed var(--border-default);
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .btn-add-tab:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
        color: var(--text-primary);
    }
</style>
