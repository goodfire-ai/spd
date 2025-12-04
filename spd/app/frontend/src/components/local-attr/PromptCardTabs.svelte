<script lang="ts">
    import type { PromptCard } from "./types";

    type Props = {
        cards: PromptCard[];
        activeCardId: string | null;
        onSelectCard: (cardId: string) => void;
        onCloseCard: (cardId: string) => void;
    };

    let { cards, activeCardId, onSelectCard, onCloseCard }: Props = $props();

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
</div>

<style>
    .card-tabs {
        display: flex;
        gap: 0.25rem;
        flex: 1;
        overflow-x: auto;
    }

    .card-tab {
        display: flex;
        align-items: center;
        background: #e0e0e0;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #616161;
        flex-shrink: 0;
    }

    .card-tab:hover {
        background: #d5d5d5;
    }

    .card-tab.active {
        background: #fff;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .card-tab-label {
        padding: 0.4rem 0.6rem;
        background: transparent;
        border: none;
        font-size: inherit;
        color: inherit;
        cursor: pointer;
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .card-tab-close {
        padding: 0.4rem 0.4rem;
        background: transparent;
        border: none;
        font-size: 0.85rem;
        line-height: 1;
        opacity: 0.5;
        cursor: pointer;
        color: inherit;
    }

    .card-tab-close:hover {
        opacity: 1;
    }
</style>
