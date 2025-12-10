<script lang="ts">
    type TokenValue = {
        token: string;
        value: number;
    };

    type Props = {
        items: TokenValue[];
        /** PMI scale: value at which color reaches full intensity */
        maxScale?: number;
        compact?: boolean;
    };

    let { items, maxScale = 10, compact = false }: Props = $props();

    // Background color for PMI (blue-white-red colorway)
    // Positive = blue, Zero = transparent, Negative = red
    function getPmiBg(value: number): string {
        if (value > 0) {
            const intensity = Math.min(1, value / maxScale);
            return `rgba(59, 130, 246, ${intensity})`;
        } else {
            const intensity = Math.min(1, -value / maxScale);
            return `rgba(220, 38, 38, ${intensity})`;
        }
    }
</script>

<div class="tokens" class:compact>
    {#each items as { token, value } (token)}
        <span class="token-pill" style="background: {getPmiBg(value)}" title="PMI: {value.toFixed(2)}">{token}</span>
    {/each}
</div>

<style>
    .tokens {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        background: var(--bg-elevated);
        padding: var(--space-2);
        border: 1px solid var(--border-default);
    }

    .tokens.compact {
        font-size: var(--text-xs);
        gap: var(--space-1);
        padding: var(--space-1);
    }

    .token-pill {
        padding: 1px 4px;
        border-radius: 3px;
        white-space: pre;
        cursor: default;
        position: relative;
    }

    .token-pill::after {
        content: attr(title);
        position: absolute;
        bottom: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: 2px 6px;
        font-size: var(--text-xs);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 100;
        border-radius: 3px;
    }

    .token-pill:hover::after {
        opacity: 1;
    }
</style>
