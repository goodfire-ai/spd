<script lang="ts">
    import type { ActivationContext } from "$lib/api";
    import TokenHighlights from "$lib/components/TokenHighlights.svelte";

    export let componentIdx: number;
    export let examples: ActivationContext[];
</script>

<div class="activation-contexts">
    <div class="header">
        <h4>Component {componentIdx}</h4>
        <span class="example-count">{examples.length} examples</span>
    </div>

    <div class="examples-container">
        {#each examples as context, i}
            <div class="example-item">
                <strong>{i + 1}.</strong>
                <TokenHighlights
                    rawText={context.raw_text}
                    offsetMapping={context.offset_mapping}
                    tokenCiValues={context.token_ci_values}
                    activePosition={context.active_position}
                />
            </div>
        {/each}
    </div>
</div>

<style>
    .activation-contexts {
        padding: 1rem;
        background: var(--background-color, #fff);
        border-radius: 6px;
        border: 1px solid #e0e0e0;
    }

    .header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .header h4 {
        margin: 0;
        font-size: 1rem;
        color: #333;
    }

    .example-count {
        font-size: 0.85rem;
        color: #999;
        margin-left: auto;
    }

    .examples-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        overflow-y: auto;
    }

    .example-item {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.8;
        color: #333;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 4px;
        border: 1px solid #e9ecef;
    }

</style>
