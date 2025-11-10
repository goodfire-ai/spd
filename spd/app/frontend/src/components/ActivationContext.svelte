<script lang="ts">
    import type { ActivationContext } from "../lib/api";
    import TokenHighlights from "./TokenHighlights.svelte";

    interface Props {
        example: ActivationContext;
    }

    let { example }: Props = $props();
</script>

<div class="example-item">
    <TokenHighlights
        tokenStrings={example.token_strings}
        tokenCiValues={example.token_ci_values}
        activePosition={example.active_position}
    />
</div>

<style>
    .example-item {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.8;
        color: #333;
        padding: 0.5rem;
        /* background: white; */
        /* border-radius: 4px;
        border: 1px solid #e9ecef; */
        overflow: visible;
    }
</style>
