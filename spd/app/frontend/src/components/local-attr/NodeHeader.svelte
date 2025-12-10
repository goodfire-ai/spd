<script lang="ts">
    type Props = {
        layer: string;
        seqIdx: number;
        cIdx: number;
        token?: string;
        onClose?: () => void;
    };

    let { layer, seqIdx, cIdx, token, onClose }: Props = $props();
</script>

<div class="node-header">
    <div class="node-info">
        <strong>{layer}:{seqIdx}:{cIdx}</strong>
        {#if token}
            <span class="token-preview">"{token}"</span>
        {/if}
    </div>
    {#if onClose}
        <button class="close-btn" onclick={onClose}>✕</button>
    {/if}
</div>

<style>
    .node-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: var(--space-2);
    }

    .node-info {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .node-header strong {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-primary);
        font-weight: 600;
    }

    .token-preview {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .close-btn {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
        padding: var(--space-1) var(--space-2);
    }

    .close-btn:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }
</style>
