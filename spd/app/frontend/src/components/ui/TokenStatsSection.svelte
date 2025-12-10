<script lang="ts">
    import TokenPillList from "./TokenPillList.svelte";
    import StatusText from "./StatusText.svelte";

    type TokenValue = {
        token: string;
        value: number;
    };

    type TokenList = {
        title: string;
        mathNotation?: string;
        items: TokenValue[];
    };

    type Props = {
        sectionTitle: string;
        sectionSubtitle?: string;
        lists: TokenList[];
        loading: boolean;
        compact?: boolean;
    };

    let { sectionTitle, sectionSubtitle, lists, loading, compact = false }: Props = $props();

    const hasData = $derived(lists.some((list) => list.items.length > 0));
</script>

<div class="token-stats-section" class:compact>
    <p class="token-stats-title">
        {sectionTitle}
        {#if sectionSubtitle}
            <span class="math-notation">{sectionSubtitle}</span>
        {/if}
    </p>
    {#if hasData}
        <div class="token-stats">
            {#each lists as list (list.title + (list.mathNotation ?? ""))}
                {#if list.items.length > 0}
                    <div class="token-list">
                        <h5>
                            {list.title}
                            {#if list.mathNotation}
                                <span class="math-notation">{list.mathNotation}</span>
                            {/if}
                        </h5>
                        <TokenPillList items={list.items} {compact} />
                    </div>
                {/if}
            {/each}
        </div>
    {:else if loading}
        <StatusText variant="muted" {compact}>Loading...</StatusText>
    {:else}
        <StatusText variant="muted" {compact}>No data available.</StatusText>
    {/if}
</div>

<style>
    .token-stats-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .token-stats-title {
        margin: 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .compact .token-stats-title {
        font-size: var(--text-xs);
    }

    .token-stats {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .token-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .token-list h5 {
        margin: 0;
        font-size: var(--text-xs);
        font-family: var(--font-sans);
        color: var(--text-muted);
        font-weight: 500;
    }

    .math-notation {
        font-family: var(--font-mono);
        font-style: normal;
        font-size: var(--text-xs);
        opacity: 0.7;
    }
</style>
